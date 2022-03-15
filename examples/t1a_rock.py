from digirock import WaterECL, WoodsFluid, Mineral, VRHAvg, NurCriticalPoroAdjust, GassmannRock, DeadOil, Transform
from digirock.utils.ecl import EclStandardConditions
from digirock.fluids.bw92 import wat_salinity_brine
from digirock.typing import NDArrayOrFloat, NDArrayOrInt

sal = wat_salinity_brine(EclStandardConditions.TEMP.value, EclStandardConditions.PRES.value, 1.00916)
wat = WaterECL(31.02641, 1.02, 3E-6, 0.8, sal)
oil = DeadOil(std_density=0.784905)
oil.set_pvt([1.25, 1.2, 1.15], [2.06843, 5.51581, 41.36854])
fl = WoodsFluid(["swat", "soil"], [wat, oil])

sand = Mineral(2.75, 32, 14)
clay = Mineral(2.55, 25, 8)
vrh = VRHAvg(["vsand", "vclay"], [sand, clay])
ncp = NurCriticalPoroAdjust(["poro"], vrh, 0.39)

grock = GassmannRock(ncp, vrh, fl)

class VShaleMult(Transform):
    
    _methods = ["vp", "vs", "density", "bulk_modulus", "shear_modulus"]
    
    def __init__(self, element, mult):
        super().__init__(["NTG"], element, self._methods)
        self._mult = mult
        
    def density(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.density(props, **kwargs)

    def vp(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.vp(props, **kwargs)
    
    def vs(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.vs(props, **kwargs)
    
    def bulk_modulus(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.bulk_modulus(props, **kwargs)
    
    def shear_modulus(self, props, **kwargs):
        props["vclay"] = props["NTG"] * self._mult
        return self.element.shear_modulus(props, **kwargs) 
    
rock = VShaleMult(grock, 0.5)