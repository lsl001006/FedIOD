import enum 
  
class TrainingArgumentsEnum(enum.Enum):
    Knowledge_Distribution=(0.9)
    Data_Free_KD=(0.5)
    
    def __init__(self,betas_min)  :
        self.betas_min=betas_min
    
    