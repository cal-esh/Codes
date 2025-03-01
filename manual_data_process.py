import numpy as np

# initial phase angles at t=0 
gamma = np.zeros(9)

# inertia matrix / pi / 60 Hz
H = np.diag([0.2228, 0.1607, 0.1899, 0.1517, 0.1379, 0.1846, 0.1401, 0.1289, 0.1830, 2.6526])

# mechanical input power
P = np.array([1.2500, 2.4943, 3.2500, 3.1600, 2.5400, 3.2500, 2.8000, 2.7000, 4.1500, 5.0000])
Q = np.array([1.61762, 2.16974, 2.06965, 1.08293, 1.66688, 2.10661, 1.00165, -0.0136945, 0.217327, -1.715326])

# damping matrix
D = np.diag([0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050])

# terminal voltages 
E = np.abs([complex(1.0568, 0.0204), complex(0.9974, 0.1770), complex(0.9842, 0.1993), complex(0.9804, 0.1809), complex(1.0855, 0.3753), complex(1.0569, 0.2160), complex(1.0423, 0.2158), complex(0.9742, 0.1817), complex(0.9352, 0.3043), complex(1.0205, -0.0431)])

# network impedance matrix G+B*i (pre fault)
impedance_matrix = np.array([ [complex(0.7239, -15.0009), complex(0.2080 ,0.9484),  complex(0.2536, 1.2183),  complex(0.2565, 1.2209),  complex(0.1033, 0.4161),  complex(0.2348, 1.0950),  complex(0.1950, 0.9237),  complex(0.0670, 2.9064),  complex(0.1961, 1.5928),  complex(0.6099, 4.8881)],
                 [complex(0.2080, 0.9484),   complex(0.2519, -9.1440), complex(0.2603, 1.8170),  complex(0.1657, 0.6891),  complex(0.0655, 0.2346),  complex(0.1513, 0.6180),  complex(0.1259, 0.5214),  complex(0.0916, 0.5287),  complex(0.1150, 0.4400),  complex(0.4159, 2.7502)],
                 [complex(0.2536, 1.2183),   complex(0.2603, 1.8170),  complex(0.3870, -10.9096), complex(0.2142, 0.9763),  complex(0.0857, 0.3326),  complex(0.1959, 0.8756),  complex(0.1629, 0.7386),  complex(0.1144, 0.6848),  complex(0.1471, 0.5888),  complex(0.4569, 2.9961)],
                 [complex(0.2565, 1.2209),   complex(0.1657, 0.6891),  complex(0.2142, 0.9763),  complex(0.8131, -12.0737), complex(0.2843, 1.9774),  complex(0.3178, 1.7507),  complex(0.2633, 1.4766),  complex(0.1608, 0.7478),  complex(0.2104, 0.8320),  complex(0.3469, 1.6513)],
                 [complex(0.1033, 0.4161),   complex(0.0655, 0.2346),  complex(0.0857, 0.3326),  complex(0.2843, 1.9774),  complex(0.1964, -5.5114), complex(0.1309, 0.5973),  complex(0.1088, 0.5038),  complex(0.0645, 0.2548),  complex(0.0826, 0.2831),  complex(0.1397, 0.5628)],
                 [complex(0.2348, 1.0950),   complex(0.1513, 0.6180),  complex(0.1959, 0.8756),  complex(0.3178, 1.7507),  complex(0.1309, 0.5973),  complex(0.4550, -11.1674), complex(0.3366, 3.1985),  complex(0.1471, 0.6707),  complex(0.1920, 0.7461),  complex(0.3175, 1.4810)],
                 [complex(0.1950, 0.9237),   complex(0.1259, 0.5214),  complex(0.1629, 0.7386),  complex(0.2633, 1.4766),  complex(0.1088, 0.5038),  complex(0.3366, 3.1985),  complex(0.4039, -9.6140), complex(0.1223, 0.5657),  complex(0.1599, 0.6294),  complex(0.2638, 1.2493)],
                 [complex(0.0670, 2.9064),   complex(0.0916, 0.5287),  complex(0.1144, 0.6848),  complex(0.1608, 0.7478),  complex(0.0645, 0.2548),  complex(0.1471, 0.6707),  complex(0.1223, 0.5657),  complex(0.6650, -10.0393), complex(0.3225, 1.2618),  complex(0.0991, 2.5318)],
                 [complex(0.1961, 1.5928),   complex(0.1150, 0.4400),  complex(0.1471, 0.5888),  complex(0.2104, 0.8320),  complex(0.0826, 0.2831),  complex(0.1920, 0.7461),  complex(0.1599, 0.6294),  complex(0.3225, 1.2618),  complex(0.9403, -7.5882), complex(0.2377, 1.5792)],
                 [complex(0.6099, 4.8881),   complex(0.4159, 2.7502),  complex(0.4569, 2.9961),  complex(0.3469, 1.6513),  complex(0.1397, 0.5628),  complex(0.3175, 1.4810),  complex(0.2638, 1.2493),  complex(0.0991, 2.5318),  complex(0.2377, 1.5792),  complex(5.9222, -18.6157)] ])
      
   
# network impedance matrix (fault on)
impedance_matrix_faulted = np.array([ [complex(0.5383, -15.7638), complex(0.0901, 0.5182),  complex(0.0994, 0.6084),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(-0.0490, 2.4392), complex(0.0472, 1.0736),  complex(0.3589, 3.8563)],
                  [complex(0.0901, 0.5182),   complex(0.1779, -9.3864), complex(0.1628, 1.4731),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0180, 0.2653),  complex(0.0219, 0.1476),  complex(0.2564, 2.1683)],
                  [complex(0.0994, 0.6084),   complex(0.1628, 1.4731),  complex(0.2591, -11.3971), complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0181, 0.3113),  complex(0.0241, 0.1739),  complex(0.2483, 2.1712)],
                  [complex(0.0000, 0.0000),   complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.4671, -14.0254), complex(0.1411, 1.3115),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000)],
                  [complex(0.0000, 0.0000),   complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.1411, 1.3115),  complex(0.1389, -5.7383), complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000)],
                  [complex(0.0000, 0.0000),   complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.1633, -12.7378), complex(0.0947, 1.8739),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000)],
                  [complex(0.0000, 0.0000),   complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0947, 1.8739),  complex(0.2035, -10.7312), complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000)],
                  [complex(-0.0490, 2.4392),  complex(0.0180, 0.2653),  complex(0.0181, 0.3113),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.5925, -10.3254), complex(0.2297, 0.9440),  complex(-0.0579, 1.8999)],
                  [complex(0.0472, 1.0736),   complex(0.0219, 0.1476),  complex(0.0241, 0.1739),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.2297, 0.9440),  complex(0.8235, -7.9409), complex(0.0363, 0.8770)],
                  [complex(0.3589, 3.8563),   complex(0.2564, 2.1683),  complex(0.2483, 2.1712),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(0.0000, 0.0000),  complex(-0.0579, 1.8999), complex(0.0363, 0.8770),  complex(5.5826, -20.0113)] ])
      
