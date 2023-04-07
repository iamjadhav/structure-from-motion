import unittest
from LoadData import load_dataset


class TestLoadData(unittest.TestCase):
    '''
    To testing the loading of dataset images
    '''
    def test_images_total(self):
        '''
        Test for total number of images/if all are read properly
        '''
        total = 6
        result = load_dataset('Data/', 6)
        self.assertEqual(len(result), total)

    def test_image_types(self):
        '''
        Test for verifying image types and making sure they are not corrupt
        '''
        image_list = load_dataset('Data/', 6)
        for file in image_list:
            self.assertIsNotNone(type(file))


if __name__ == '__main__':
    unittest.main()
