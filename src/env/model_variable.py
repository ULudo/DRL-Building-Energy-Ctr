from typing import Callable
import numpy as np

class ModelVariable:
    """Class for specifying FMU variables.
    """

    def __init__(
        self,
        name:str,
        value:any=.0,
        unit:str="",
        info:str="",
        conv_calc:Callable=None,
        bounds:list=None,
        take_delta:bool=False,
        type:str="fmu"
    ) -> None:
        """
        Args:
            name: Name of variable.
            value: Value of variable.
            unit: Physical unit of variable.
            info: Description of variable.
            conv_calc: Function for converting FMU outputs to the desired unit.
            bounds: Upper lower bound of variable (used for normalization).
            take_delta: Wether to take as value the delta between the previous and the current FMU output.
            type: Specifies if it is a FMU type variable or some other type.
        """
        assert bounds is None or len(bounds) == 2, "Bound should only contain two values: [lower bound, upper bound]"

        self.name = name
        self.value = value
        self.unit = unit
        self.info = info
        self.conv_calc = (lambda x: x) if conv_calc is None else conv_calc
        self.bounds = bounds
        self.take_delta = take_delta
        self.type = type
    
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.name == other.name and \
                   self.value == other.value and \
                   self.unit == other.unit and \
                   self.info == other.info and \
                   self.conv_calc == other.conv_calc and \
                   self.bounds == other.bounds and \
                   self.take_delta == other.take_delta and \
                   self.type == other.type
        else:
            return False
    
    
    def normalize(self, value):
        """Normalizes the value to [-1, 1]. Values that are not in between the bounds
        get clipped.
        Args - value: Value to normalize.
        """
        if self.bounds is None:
            return value
        min_val, max_val = self.bounds[0], self.bounds[1]
        norm = np.clip(value, min_val, max_val)
        norm = (norm - min_val) / (max_val - min_val) * 2 - 1
        return norm
    
    
    def rescale(self, value):
        """Rescales the value by using the bounds.
        Args - value: Value to rescale.
        """
        if self.bounds is None:
            return value
        min_val, max_val = self.bounds[0], self.bounds[1]
        denorm = 0.5 * (value + 1) * (max_val - min_val) + min_val
        return denorm
