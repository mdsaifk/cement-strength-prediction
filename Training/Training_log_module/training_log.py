import logging as lg

class train_logging:
    def __init__(self,filename):
        self.filename=filename
        self.lg=lg
        
    def true_log(self,message):
        try:
            self.lg.basicConfig(filename=self.filename,level=lg.INFO,format= '%(levelname)s-%(asctime)s-%(message)s', datefmt='%d/%m/%Y %H:%M:%S')
            return self.lg.info(message)
        finally:
            self.lg.shutdown()
            
    def error_log(self,message):
        try:
            self.lg.basicConfig(filename=self.filename,level=lg.INFO,format= '%(levelname)s-%(asctime)s-%(message)s', datefmt='%d/%m/%Y %H:%M:%S')
            return self.lg.info(message)
        finally:
            self.lg.shutdown()

