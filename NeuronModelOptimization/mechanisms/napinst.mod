TITLE Sodium persistent current for Golomb et al., J Neurophysiol 96:1912-1926, 2006
COMMENT
Implemented by Daniele Linaro, 2014 (danielelinaro@gmail.com)
ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

NEURON {
    SUFFIX napinst
    USEION na READ ena WRITE ina
    RANGE gbar, gna
}
UNITS { 
    (mV) = (millivolt) 
    (mA) = (milliamp) 
} 
PARAMETER { 
    gbar = 0.0 	(mho/cm2)  : in the paper, 0 <= gbar <= 0.00041
    ena = 55.0      (mV)  
    v               (mV)
    thetap = -47.0  (mV)   : in the paper, -47 <= thetap <= -41
    sigmap = 3.0    (mV)
}
ASSIGNED {
    gna             (mho/cm2)
    ina 		(mA/cm2) 
    pinf 		(1)
}
STATE {
    foo
}
BREAKPOINT { 
    SOLVE states METHOD cnexp
    gna = gbar*pinf
    ina = gna * (v - ena) 
}
UNITSOFF 
INITIAL { 
    settables(v)
    foo = 0
}
DERIVATIVE states { 
    settables(v) 
    foo' = 0
}
PROCEDURE settables(v) { 
    TABLE pinf FROM -120 TO 40 WITH 641
    pinf = 1. / (1. + exp(-(v-thetap)/sigmap))
}
UNITSON
