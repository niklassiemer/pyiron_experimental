import matplotlib.pyplot as plt
from pyiron_base import TemplateJob
from temmeta import data_io as dio
from temmeta import image_filters as imf
from temmeta.plottingtools import get_scalebar
from pystem.stemsegmentation import segmentationSTEM



class pySTEMTEMMETAJob(TemplateJob):
    def __init__(self, project, job_name):
        super(pySTEMTEMMETAJob, self).__init__(project, job_name) 
        self.input['file_name'] = ''
        self.input['n_patterns'] = 2
        self.input['patch_x'] = 20
        self.input['patch_y'] = 20
        self.input['window_x'] = 20
        self.input['window_y'] = 20
        self.input['step'] = 20
        self.input['upsampling'] = True
        self._python_only_job = True
        self._image = None
        self._vec = []
        self._profile = []
    
    @property
    def file_name(self):
        return self.input['file_name']
    
    @file_name.setter
    def file_name(self, file_name):
        self.input['file_name'] = file_name
        self._image = self.create_tem_image()
    
    def create_tem_image(self):
        emd1 = dio.EMDFile(self.input['file_name'])
        return emd1.get_dataset("Image", "6fdbde41eecc4375b45cd86bd2be17c0")
    
    def plot(self, labels=False, alpha=0.5):
        av = self._image.average
        ax, im = plot_image(img=av, dpi=50)
        if labels:
            labels = self["output/generic/segmentation_labels"]
            if labels is not None: 
                ax.imshow(labels, alpha=alpha)
            else: 
                raise ValueError()
    
    def perform_segmentation(self,image):
        seg = segmentationSTEM(
            n_patterns=self.input['n_patterns'],
            window_x=self.input['window_x'],
            window_y=self.input['window_y'],
            patch_x=self.input['patch_x'],
            patch_y=self.input['patch_y'],
            step=self.input['step'],
            upsampling=self.input['upsampling'])
        labels = seg.perform_clustering(image)
        return labels

    def run_static(self):
        av = self._image.average
        with self.project_hdf5.open("output/generic") as h5out: 
            h5out["segmentation_labels"] = self.perform_segmentation(av.data) 
        self.status.finished = True

        
def plot_array(imgdata, pixelsize=1., pixelunit="", scale_bar=True,
               show_fig=True, width=15, dpi=None,
               sb_settings={"location": 'lower right',
                            "color": 'k',
                            "length_fraction": 0.15,
                            "font_properties": {"size": 12}},
               imshow_kwargs={"cmap": "Greys_r"}):
    '''
    Plot a 2D numpy array as an image.
    A scale-bar can be included.
    Parameters
    ----------
    imgdata : array-like, 2D
        the image frame
    pixelsize : float, optional
        the scale size of one pixel
    pixelunit : str, optional
        the unit in which pixelsize is expressed
    scale_bar : bool, optional
        whether to add a scale bar to the image. Defaults to True.
    show_fig : bool, optional
        whether to show the figure. Defaults to True.
    width : float, optional
        width (in cm) of the plot. Default is 15 cm
    dpi : int, optional
        alternative to width. dots-per-inch can give an indication of size
        if the image is printed. Overrides width.
    sb_settings : dict, optional
        key word args passed to the scale bar function. Defaults are:
        {"location":'lower right', "color" : 'k', "length_fraction" : 0.15,
         "font_properties": {"size": 40}}
        See: <https://pypi.org/project/matplotlib-scalebar/>
    imshow_kwargs : dict, optional
        optional formating arguments passed to the pyplot.imshow function.
        Defaults are: {"cmap": "Greys_r"}
    Returns
    -------
    ax : matplotlib Axis object
    im : the image plot object
    '''
    # initialize the figure and axes objects
    if not show_fig:
        plt.ioff()
    if dpi is not None:
        fig = plt.figure(frameon=False,
                         figsize=(imgdata.shape[1]/dpi, imgdata.shape[0]/dpi))
    else:
        # change cm units into inches
        width = width*0.3937008
        height = width/imgdata.shape[1]*imgdata.shape[0]
        fig = plt.figure(frameon=False,
                         figsize=(width, height))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot the figure on the axes
    im = ax.imshow(imgdata, **imshow_kwargs)

    if scale_bar:
        # get scale bar info from metadata
        px = pixelsize
        unit = pixelunit
        # check the units and adjust sb accordingly
        scalebar = get_scalebar(px, unit, sb_settings)
        plt.gca().add_artist(scalebar)
    # if show_fig:
    #     plt.show()
    # else:
    #     plt.close()
    return ax, im


def plot_image(img, **kwargs):
    """
    Wrapper for plot_array using a GeneralImage object directly
    """
    ax, im = plot_array(img.data, pixelsize=img.pixelsize,
                        pixelunit=img.pixelunit, **kwargs)
    return ax, im
