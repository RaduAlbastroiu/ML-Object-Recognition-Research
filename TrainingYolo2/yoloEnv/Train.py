from darkflow.net.build import TFNet

options = {'model':'cfg/yolo-train.cfg',
          'load':'yolov2-tiny.weights', 
          'epoch':10,
          'train':True,
          'annotation': '/Users/radualbastroiu/Documents/My_projects/Licenta/ML-Object-Recognition/data/CarDownloadImagesLabeled/annotations/',
          'annotation': '/Users/radualbastroiu/Documents/My_projects/Licenta/ML-Object-Recognition/data/CarDownloadImagesLabeled/images/'
          }

tfnet = TFNet(options)
tfnet.train()