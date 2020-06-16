import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

class microscope_info:
    """General class that contains all the information how to treat the microscopy data
    for Dataset 1 and Dataset 2, described in original Thesis text."""
    extension='.tif'
    folder_Z1_original = './data_Z1/tiffs/'
    folder_ultra_original = './data_LightSheet/'
    dtype = np.uint8
    
    @staticmethod
    def _sort_rule_Z1(name):
        return int(name.split(' ')[-1].split('.')[0])
    
    @staticmethod
    def _sort_rule_light_sheet(name):
        return int( name.split('.')[0].split('section')[-1] )
    
    @staticmethod
    def preprocess(image):
        return image
    
    def __init__(self):
        self.name = None
        self.folder = None
        self.sorting_rule = None


        
class z1(microscope_info):
    def __init__(self):
        super().__init__()
        self.name = 'z1'
        self.folder = self.folder_Z1_original
        self.sorting_rule = self._sort_rule_Z1
        
    @staticmethod
    def preprocess(image):
        image = image[:,:,0]
        image = image.T
        return image.astype(np.uint8)


    
class ls(microscope_info):
    def __init__(self):
        self.name = 'ls'
        self.folder = self.folder_ultra_original
        self.sorting_rule = self._sort_rule_light_sheet
    
    @staticmethod
    def preprocess(image):
        image = 255*image.astype(np.float32)/65536
        return image.astype(np.uint8)
    

class stack3D:  
    """Class that contains all the necessary functions to load, 
    visualize and crop the 3D image of microscopy data"""
    _preview_height = 500
    _preview_cmap = 'gray'
    
    def _load_slice(self, img_name):
        img_slice = cv2.imread(os.path.join(self._folder, img_name), -1)     
        img_slice = self._image_preprocess(img_slice)
        return img_slice
        
    def _load_example(self, idx=None):
        if idx:
            img_name = self.slices[idx]
            print('example:', img_name)
        else:
            img_name = np.random.choice(self.slices)
            print('random example:', img_name)
            
        img_slice = self._load_slice(img_name)
        ndims, x_y_shape, data_type = img_slice.ndim, img_slice.shape[:2], img_slice.dtype
        return img_slice, ndims, x_y_shape, data_type
    
    def preview(self, image, axes, resize=True):
        if resize:
            img_res = imutils.resize(image, height=self._preview_height)
        else:
            img_res = image
        axes.imshow(img_res, cmap=self._preview_cmap)
        axes.set_xlim(0, img_res.shape[1])
        axes.set_ylim(img_res.shape[0], 0)
        x_ticks = axes.get_xticks()
        y_ticks = axes.get_yticks()
        
        x_ticks_labels = np.linspace(0,image.shape[1],len(x_ticks), dtype=int)
        y_ticks_labels = np.linspace(0,image.shape[0],len(y_ticks), dtype=int)
        axes.set_xticklabels(x_ticks_labels)
        axes.set_yticklabels(y_ticks_labels)
        axes.set_xlabel('x', fontsize=14)
        axes.set_ylabel('y', fontsize=14)
        return axes
    
    @staticmethod
    def _convert_memory_values(value):
        v_type = 'bits'
        if value > 8:
            value /= 8
            v_type = 'bytes'
        if value > 1024:
            value /= 1024
            v_type = 'kilobytes'
        if value > 1024:
            value /= 1024
            v_type = 'megabytes'
        if value > 1024:
            value /= 1024
            v_type = 'gigabytes'
        return round(value,2), v_type
    
    def _estimate_memory(self, size=None):
        if self._loading_dtype == np.uint8:
            bits = 8
        elif self._loading_dtype == np.float32:
            bits = 32
            
        if size is None:
            slice_memory = self._slice_shape[0]*self._slice_shape[1]*bits
            full_stack_memory = len(self.slices)*slice_memory
            return self._convert_memory_values(slice_memory), self._convert_memory_values(full_stack_memory)   
        else:
            mem_bits = size*bits
            return self._convert_memory_values(mem_bits)
    
    
    def __init__(self, microscope):
        self._folder = microscope.folder
        self._image_preprocess = microscope.preprocess
        self._extension = microscope.extension
        self._loading_dtype = microscope.dtype
        
        # Read information about all images available
        # Sort images to left only with correct extension
        # Sort images in growing order
        names_list = os.listdir(self._folder)
        names_list = [img for img in names_list if img.endswith(self._extension)]
        names_list.sort(key = microscope.sorting_rule)
        self.slices = names_list
        
        # Read data about stack 
        example_slice, self._ndim, self._slice_shape, self._dtype = self._load_example()
        memory_results = self._estimate_memory()
        
        # Visualize
        print('Num of slices:', len(self.slices), ', slice shape:', self._slice_shape, ', initial dtype:', self._dtype)
        print('Slice memory:', memory_results[0][0],memory_results[0][1]+';', 
              'Brain memory:', memory_results[1][0], memory_results[1][1])
        fig, axes = plt.subplots(1,1)
        fig.set_figheight(6)
        fig.set_figwidth(8)
        axes = self.preview(example_slice, axes)
        plt.show()
        del example_slice
        
    def get_sample(self, coordinates:tuple=None):
        # x - along width (slice.shape[1])
        # y - along height (slice.shape[0])
        # z - along slices (self.slices)
        if coordinates is not None:
            ((x_0,x_1), (y_0,y_1), (z_0,z_1)) = coordinates
        else:
            size = 250
            x_0 = np.random.randint( self._slice_shape[1]*0.1, min(self._slice_shape[1]*0.9, self._slice_shape[1]-size) )
            x_1 = x_0 + size
            
            y_0 = np.random.randint( self._slice_shape[0]*0.1, min(self._slice_shape[0]*0.9, self._slice_shape[0]-size) )
            y_1 = y_0 + size
            
            total_slices = len(self.slices)
            z_size = 5
            z_0 = np.random.randint( int(total_slices*0.2), int(total_slices*0.7))
            z_1 = z_0 + z_size
            coordinates = ((x_0,x_1), (y_0,y_1), (z_0,z_1))
        
        print('x:',coordinates[0], 'y:',coordinates[1], 'z:',coordinates[2])
        
        if z_1 == z_0:
            # One 2D slcie
            slices_to_load = self.slices[z_0:z_0+1]
        else:
            # Several slices
            slices_to_load = self.slices[z_0:z_1]
        memory_results = self._estimate_memory( (y_1-y_0)*(x_1-x_0)*(z_1-z_0) )
        if memory_results[1] in ['bits', 'bytes', 'kilobytes', 'megabytes']:
            pass
        else:
            raise Exception('The asked sample is too large to be loaded')
        
        sample = np.zeros((y_1-y_0, x_1-x_0, z_1-z_0), dtype = self._loading_dtype)
        for z_idx, img_name in enumerate( slices_to_load ):
            tmp_slice = self._load_slice(img_name)
            sample[:,:,z_idx] = tmp_slice[y_0:y_1, x_0:x_1]
        return sample
    
    def projections(self, image, transpose=True, flip=False, resize=False):
        self.max_projections(image, transpose=transpose, flip=flip, resize=resize)
    
    def max_projections(self, image, transpose=True, flip=False, resize=False):
        if len(image.shape) < 3:
            raise Exception('Can not build projections for non-3D image')
        proj_1 = np.max(image, 0)
        proj_2 = np.max(image, 1)
        proj_3 = np.max(image, 2)
        if transpose:
            proj_1 = np.transpose(proj_1, (1,0))
            proj_2 = np.transpose(proj_2, (1,0))
        if flip:
            proj_1 = np.flip(proj_1, 0)
            proj_2 = np.flip(proj_2, 0)
        
        
        fig, axes = plt.subplots(2,2)
        fig.set_figheight(10)
        fig.set_figwidth(14)
        axes[0][0] = self.preview(proj_1, axes[0][0], resize)
        axes[0][0].set_ylabel('z', fontsize=14)
        
        axes[0][1] = self.preview(proj_2, axes[0][1], resize)
        axes[0][1].set_ylabel('z', fontsize=14)
        axes[0][1].set_xlabel('y', fontsize=14)
        
        axes[1][0] = self.preview(proj_3, axes[1][0], resize)
        axes[1][1].axis('off')
        plt.show()
        
    def idx_projections(self, image, slice_x, slice_y, slice_z, transpose=True, flip=False, resize=False):
        if len(image.shape) < 3:
            raise Exception('Can not build projections for non-3D image')
        proj_1 = image[slice_x,:,:]
        proj_2 = image[:,slice_y,:]
        proj_3 = image[:,:,slice_z]
        if transpose:
            proj_1 = np.transpose(proj_1, (1,0))
            proj_2 = np.transpose(proj_2, (1,0))
        if flip:
            proj_1 = np.flip(proj_1, 0)
            proj_2 = np.flip(proj_2, 0)
        
        
        fig, axes = plt.subplots(2,2)
        fig.set_figheight(10)
        fig.set_figwidth(14)
        axes[0][0] = self.preview(proj_1, axes[0][0], resize)
        axes[0][0].set_ylabel('z', fontsize=14)
        
        axes[0][1] = self.preview(proj_2, axes[0][1], resize)
        axes[0][1].set_ylabel('z', fontsize=14)
        axes[0][1].set_xlabel('y', fontsize=14)
        
        axes[1][0] = self.preview(proj_3, axes[1][0], resize)
        axes[1][1].axis('off')
        plt.show()