import numpy as np
from scipy.sparse import csr_matrix
import numexpr as ne

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from hcipy.optics.optical_element import OpticalElement
# from hcipy.field import make_uniform_grid, evaluate_supersampled
from hcipy.field import evaluate_supersampled
from SamUtil import make_uniform_grid

from hcipy.mode_basis import ModeBasis, make_gaussian_pokes
from hcipy.interpolation import make_linear_interpolator_separated
from hcipy.util import read_fits
from hcipy.field.coordinates import UnstructuredCoords
from hcipy.field.cartesian_grid import CartesianGrid

def make_circular_actuator_positions(num_rings, points_first_ring, actuator_spacing, 
                                     x_tilt=0, y_tilt=0, z_tilt=0):
    ''' 
    Defines a circular pupil by distributing actuators in concentric rings with interleaved positions.

    Each successive ring is rotated by half the angular spacing of the previous ring to ensure interleaving.

    Parameters
    ----------
    num_rings : integer
        The number of concentric rings (not counting the center actuator).
    points_first_ring : integer
        The number of actuator points on the first ring.
    actuator_spacing : scalar
        The spacing (radial gap) between consecutive rings.
    x_tilt : scalar
        The tilt of the deformable mirror around the x-axis in radians.
    y_tilt : scalar
        The tilt of the deformable mirror around the y-axis in radians.
    z_tilt : scalar
        The rotation (or tilt) of the deformable mirror around the z-axis in radians.

    Returns
    -------
    Grid
        The actuator positions arranged in a circular interleaved layout.
    '''
    points = []
    # Add the center actuator.
    points.append((0.0, 0.0))
    
    prev_num_act = points_first_ring

    for ring in range(1, num_rings + 1):
        r = ring * actuator_spacing
        num_act = ring * points_first_ring
        
        # Angular offset for interleaving
        offset = (np.pi / prev_num_act) if ring > 1 else 0

        for i in range(num_act):
            theta = offset + 2 * np.pi * i / num_act
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append((x, y))
        
        prev_num_act = num_act  # Update for next ring

    points = np.array(points)
    
    # Apply tilt scaling (x scaled by cos(y_tilt), y scaled by cos(x_tilt))
    points[:, 0] *= np.cos(y_tilt)
    points[:, 1] *= np.cos(x_tilt)
    
    # Apply a rotation around the z-axis if needed.
    if z_tilt != 0:
        c = np.cos(z_tilt)
        s = np.sin(z_tilt)
        x_rot = c * points[:, 0] - s * points[:, 1]
        y_rot = s * points[:, 0] + c * points[:, 1]
        points[:, 0] = x_rot
        points[:, 1] = y_rot
    
    # Convert the list of points to a Grid.
    from hcipy.field.coordinates import UnstructuredCoords
    from hcipy.field.cartesian_grid import CartesianGrid
    grid = CartesianGrid(UnstructuredCoords([points[:, 0], points[:, 1]]))
    return grid



def make_triangular_actuator_positions(num_rings, actuator_spacing,
                                       x_tilt=0, y_tilt=0, z_tilt=0):
    '''
    Generate actuator positions on a triangular lattice via hexagonal rings around the center.

    Uses basis vectors:
      v1 = (actuator_spacing, 0)
      v2 = (actuator_spacing/2, sqrt(3)/2 * actuator_spacing)
    Each ring k contains all lattice points with hex distance = k (6k points).
    '''
    # Lattice basis vectors
    v1 = np.array([actuator_spacing, 0.0])
    v2 = np.array([actuator_spacing/2, actuator_spacing * np.sqrt(3)/2])

    points = [(0.0, 0.0)]
    # Build hex rings
    for k in range(1, num_rings + 1):
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                if max(abs(i), abs(j), abs(-i - j)) == k:
                    pos = i * v1 + j * v2
                    points.append((pos[0], pos[1]))

    points = np.array(points)
    # Tilts
    points[:, 0] *= np.cos(y_tilt)
    points[:, 1] *= np.cos(x_tilt)
    if z_tilt != 0:
        c, s = np.cos(z_tilt), np.sin(z_tilt)
        x_rot = c * points[:, 0] - s * points[:, 1]
        y_rot = s * points[:, 0] + c * points[:, 1]
        points[:, 0], points[:, 1] = x_rot, y_rot

    return CartesianGrid(UnstructuredCoords([points[:, 0], points[:, 1]]))


def make_gaussian_influence_functions(pupil_grid, num_rings, points_first_ring, 
                                      actuator_spacing, circular_layout=True, 
                                      triangle_layout=False, crosstalk=0.8, cutoff=3, 
                                      x_tilt=0, y_tilt=0, z_tilt=0, oversampling=1):
    ''' 
    Create influence functions with a Gaussian profile.
    
    For a circular layout, actuators are distributed in concentric rings (using make_circular_actuator_positions)
    and only those inside a circular aperture (with radius equal to the outermost ring) are kept.
    
    For a triangle layout, actuators are distributed in rings that form equilateral triangles.
    In that case, the first ring must have exactly 3 actuators (one at each vertex), i.e., points_first_ring must equal 3.
    
    Parameters
    ----------
    pupil_grid : Grid
        The grid on which to calculate the influence functions.
    num_rings : integer
        For circular layout: the number of concentric rings (excluding the center).
        For triangle layout: the number of triangular rings (excluding the center).
    points_first_ring : integer
        For circular layout: the number of actuator points on the first ring.
        For triangle layout: this must be 3.
    actuator_spacing : scalar
        The spacing (radial gap for circular or side length for triangle rings).
    crosstalk : scalar
        The crosstalk value (influence at a nearest-neighbour actuator).
    cutoff : scalar
        The cutoff distance (in units of the actuator_spacing/σ) beyond which the 
        influence function is truncated.
    x_tilt, y_tilt, z_tilt : scalar
        Tilts/rotations of the DM (in radians).
    oversampling : integer
        The oversampling factor when creating the Gaussian.
    
    Returns
    -------
    ModeBasis
        The Gaussian influence functions for the valid actuators.
    '''
                
    if circular_layout:
        actuator_positions = make_circular_actuator_positions(
            num_rings, points_first_ring, actuator_spacing,
            x_tilt, y_tilt, z_tilt)
    elif triangle_layout:
        actuator_positions = make_triangular_actuator_positions(
            num_rings, actuator_spacing, x_tilt, y_tilt, z_tilt)
    else:
        # actuator_positions = make_actuator_positions(
        #     num_actuators_across_pupil=points_first_ring,
        #     actuator_spacing=actuator_spacing,
        #     x_tilt=x_tilt, y_tilt=y_tilt, z_tilt=z_tilt)
        print("No layout specified, using circular layout by default.")
        actuator_positions = make_circular_actuator_positions(
            num_rings, points_first_ring, actuator_spacing,
            x_tilt, y_tilt, z_tilt)

    # Filter to circular pupil
    max_radius = num_rings * actuator_spacing
    x, y = actuator_positions.points.T
    margin = actuator_spacing / 2  # Margin to include the outermost ring
    mask = np.sqrt(x**2 + y**2) <= (max_radius+margin)
    filtered_actuator_positions = actuator_positions.subset(mask)

    # Compute sigma from the actuator_spacing and crosstalk.

    sigma = actuator_spacing / np.sqrt(-2 * np.log(crosstalk))
    cutoff = actuator_spacing / sigma * cutoff


    def transform_poke(poke):
        def new_poke(grid):
            p = poke(grid.scaled(1 / np.cos([y_tilt, x_tilt])).rotated(-z_tilt))
            p /= np.cos(x_tilt) * np.cos(y_tilt)
            return p
        return new_poke

    # Generate influence functions (using a helper to create Gaussian “pokes”)
    pokes = make_gaussian_pokes(None, filtered_actuator_positions, sigma, cutoff)
    pokes = [transform_poke(p) for p in pokes]
    pokes = evaluate_supersampled(pokes, pupil_grid, oversampling, make_sparse=True)
    
    return pokes

def make_xinetics_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing, x_tilt=0, y_tilt=0, z_tilt=0):
    '''Create influence functions for a Xinetics deformable mirror.

    This function uses a The rotation of the deformable mirror will be done in the order X-Y-Z.

    Parameters
    ----------
    pupil_grid : Grid
        The grid on which to calculate the influence functions.
    num_actuators_across_pupil : integer
        The number of actuators across the pupil. The total number of actuators will be this number squared.
    actuator_spacing : scalar
        The spacing between actuators before tilting the deformable mirror.
    x_tilt : scalar
        The tilt of the deformable mirror around the x-axis in radians.
    y_tilt : scalar
        The tilt of the deformable mirror around the y-axis in radians.
    z_tilt : scalar
        The tilt of the deformable mirror around the z-axis in radians.

    Returns
    -------
    ModeBasis
        The influence functions for each of the actuators.
    '''
    actuator_positions = make_circular_actuator_positions(num_actuators_across_pupil, actuator_spacing)

    # Stretch and rotate pupil_grid to correct for tilted DM
    evaluated_grid = pupil_grid.scaled(1 / np.cos([y_tilt, x_tilt])).rotated(-z_tilt)

    # Read in actuator shape from file.
    f = files('hcipy.optics').joinpath('influence_dm5v2.fits')
    with f.open('rb') as fp:
        actuator = np.squeeze(read_fits(fp)).astype('float')
    actuator /= actuator.max()

    # Convert actuator into linear interpolator.
    actuator_grid = make_uniform_grid(actuator.shape, np.array(actuator.shape) * actuator_spacing / 10.0)
    actuator = make_linear_interpolator_separated(actuator.ravel(), actuator_grid, 0)

    def poke(p):
        res = csr_matrix(actuator(evaluated_grid.shifted(-p))) / np.cos(x_tilt) * np.cos(y_tilt)
        res.eliminate_zeros()

        return res

    return ModeBasis([poke(p) for p in actuator_positions.points], pupil_grid)

def find_illuminated_actuators(basis, aperture, power_cutoff=0.1):
    '''Find the illuminated modes.

    A subset of the modes is selected based on the aperture function and a power cutoff.

    Parameters
    ----------
    basis : ModeBasis
        The mode basis for which we want to find the illuminated modes.
    aperture : Field or array_like
        The aperture
    power_cutoff : scalar
        The minimal required power over the aperture.

    Returns
    -------
    ModeBasis
        The illuminated influence functions.
    '''
    total_power = np.sum(abs(basis._transformation_matrix)**2, axis=0)
    masked_power = np.sum(abs(basis._transformation_matrix[aperture > 0])**2, axis=0)
    illuminated_actuator_mask = masked_power >= (power_cutoff * total_power)

    return ModeBasis(basis._transformation_matrix[:, illuminated_actuator_mask], basis.grid)

class SamDeformableMirror(OpticalElement):
    '''A deformable mirror using influence functions.

    This class does not contain any temporal simulation (ie. settling time),
    and assumes that there is no crosstalk between actuators.

    Parameters
    ----------
    influence_functions : ModeBasis
        The influence function for each of the actuators.
    '''
    def __init__(self, influence_functions):
        self.influence_functions = influence_functions

        self.actuators = np.zeros(len(influence_functions))
        self._actuators_for_cached_surface = None

        self.input_grid = influence_functions.grid
        self._surface = self.input_grid.zeros()

    @property
    def num_actuators(self):
        return self._actuators.size

    @property
    def actuators(self):
        return self._actuators

    @actuators.setter
    def actuators(self, actuators):
        self._actuators = actuators

    def forward(self, wavefront):
        '''Propagate a wavefront through the deformable mirror.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The reflected wavefront.
        '''
        wf = wavefront.copy()

        variables = {'alpha': 2j * wavefront.wavenumber, 'surf': self.surface}
        wf.electric_field *= ne.evaluate('exp(alpha * surf)', local_dict=variables)

        return wf

    def backward(self, wavefront):
        '''Propagate a wavefront backwards through the deformable mirror.

        Parameters
        ----------
        wavefront : Wavefront
            The incoming wavefront.

        Returns
        -------
        Wavefront
            The reflected wavefront.
        '''
        wf = wavefront.copy()

        variables = {'alpha': -2j * wavefront.wavenumber, 'surf': self.surface}
        wf.electric_field *= ne.evaluate('exp(alpha * surf)', local_dict=variables)

        return wf

    @property
    def influence_functions(self):
        '''The influence function for each of the actuators of this deformable mirror.
        '''
        return self._influence_functions

    @influence_functions.setter
    def influence_functions(self, influence_functions):
        self._influence_functions = influence_functions
        self._actuators_for_cached_surface = None

    @property
    def surface(self):
        '''The surface of the deformable mirror in meters.
        '''
        if self._actuators_for_cached_surface is not None:
            if np.all(self.actuators == self._actuators_for_cached_surface):
                return self._surface

        self._surface = self.influence_functions.linear_combination(self.actuators)
        self._actuators_for_cached_surface = self.actuators.copy()

        return self._surface

    @property
    def opd(self):
        '''The optical path difference in meters that this deformable
        mirror induces.
        '''
        # optical path difference is zero outside the aperture
        pupil_radius = self.input_grid.extent[0] / 2  # Compute pupil radius from grid
        r = np.sqrt(self.input_grid.x**2 + self.input_grid.y**2)  # Convert to polar coordinates
        
        opd = 2 * self.surface
        opd[r > pupil_radius] = 0  # Zero out values outside the circular pupil
        
        return opd
        # return 2 * self.surface

    def random(self, rms):
        '''Set the dm actuators with random white noise of a certain rms.

        Parameters
        ----------
        rms : scalar
            The dm surface rms.
        '''
        self._actuators = np.random.randn(self._actuators.size) * rms

    def phase_for(self, wavelength):
        '''Get the phase in radians that is added to a wavefront with a specified wavelength.

        Parameters
        ----------
        wavelength : scalar
            The wavelength at which to calculate the phase deformation.

        Returns
        -------
        Field
            The calculated phase deformation.
        '''
        return 2 * self.surface * 2 * np.pi / wavelength

    def flatten(self):
        '''Flatten the DM by setting all actuators to zero.
        '''
        self._actuators = np.zeros(len(self.influence_functions))

def label_actuator_centroid_positions(influence_functions, label_format='{:d}', **text_kwargs):
    '''Display centroid positions for a set of influence functions.

    The location of each of the actuators is calculated using a weighted centroid, and
    at that location a label is written to the open Matplotlib Figure. The text can be
    modified with `label_format`, which is formatted with new-style Python formatting:
    `label_format.format(i)` where `i` is the actuator index.

    Parameters
    ----------
    influence_functions : ModeBasis
        The influence function for each actuator.
    label_format : string
        The text that will be displayed at the actuator centroid. This must be a new-style
        formattable string.

    Raises
    ------
    ValueError
        If the influence functions mode basis does not contain a grid.
    '''
    # only label actuators inside the pupil.
    import matplotlib.pyplot as plt

    if influence_functions.grid is None:
        raise ValueError('The influence functions mode basis must contain a grid to calculate centroids.')

    grid = influence_functions.grid
    x, y = grid.coords
    r = np.sqrt(x**2 + y**2)  # Convert to polar coordinates

    # Center the labels unless overridden by the user.
    kwargs = {'verticalalignment': 'center', 'horizontalalignment': 'center'}
    kwargs.update(text_kwargs)

    for i, act in enumerate(influence_functions):
        x_pos = (act * x).sum() / act.sum()
        y_pos = (act * y).sum() / act.sum()
        pos = (x_pos, y_pos)

        # Only annotate actuators within the pupil radius
        if np.sqrt(x_pos**2 + y_pos**2) <= pupil_radius:
            plt.annotate(label_format.format(i), xy=pos, **kwargs)