COMMENT
Current clamp that generates a sinusoidal current

Author: Daniele Linaro
Date: December 20, 2014
ENDCOMMENT

NEURON {
    POINT_PROCESS SineClamp
    RANGE delay, dur, amp, f, phi
    GLOBAL pi
    ELECTRODE_CURRENT i
    THREADSAFE
}

UNITS {
    (nA) = (nanoamp)
}

PARAMETER {
    delay          (ms)
    dur            (ms)
    amp            (nA)
    f              (/s)
    phi            (1)
    pi = 3.141592653589793 (1)
}

ASSIGNED {
    i      (nA)
}

INITIAL {
    i = 0
}

BREAKPOINT {
    at_time(delay)
    at_time(delay+dur)
    if (t>=delay && t<=delay+dur) {
	i = amp * sin(2*pi * (0.001)*(f*t) - phi)
    } else {
	i = 0
    }
}

