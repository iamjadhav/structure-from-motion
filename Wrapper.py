'''
Structrure-From-Motion: Pipeline to reconstruct a 3D scene from 6 stereo images.
'''

from LoadData import load_dataset, load_data

def main():
    '''
    The entire pipeline strung together
    '''
    data_folder = 'Data/'
    total_images = 6
    # Load dataset
    images = load_dataset(data_folder, total_images)
    load_data(data_folder, total_images)































if __name__ == '__main__':
    main()
