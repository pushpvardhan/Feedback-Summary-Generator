import numpy as np
import pandas as pd
from sympy.abc import x

# df = pd.read_excel("Peer Form 1.xlsx")
# # print(df.to_string())
# df1 = df.drop(['Options'], axis=1)
#
# # df1 = df1.reset_index()
# # print(df1.to_string())
# print('***********************************')
# print(df1.iloc[0, 0])
# print(df1.iloc[0, 1])
# print(df1.iloc[0, 2])
# print(df1.iloc[0, 3])
# print(df1.iloc[0, 4])
# print('***********************************')
# # print(df1.iloc[0:,7])

x = [[101, 2129, 4703, 2024, 1996, 5604, 4084, 8832, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2040, 2515, 1996, 3188, 25090, 9276, 1997, 8518, 2241, 2006, 5918, 1013, 17060, 2015, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2079, 9095, 16680, 3961, 1999, 24977, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2247, 2007, 1996, 2136, 2013, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102, 0, 0, 0], [101, 2003, 2393, 3223, 2000, 3188, 25090, 4371, 8518, 5834, 2006, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2079, 9095, 16680, 3961, 1999, 24977, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2040, 2515, 1996, 3188, 25090, 9276, 1997, 8518, 2241, 2006, 5918, 1013, 17060, 2015, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2011, 2969, 1006, 2203, 2000, 2203, 1007, 2007, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102], [101, 2129, 4703, 2079, 9095, 16680, 3961, 1999, 24977, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3446, 4824, 1997, 1996, 3800, 1013, 4432, 1013, 2449, 3289, 1997, 1996, 4031, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3446, 1996, 4824, 1997, 1996, 3800, 1013, 3289, 1997, 1996, 14184, 2499, 2006, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2024, 1996, 5604, 4084, 8832, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 4194, 1999, 23138, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3446, 4824, 1997, 1996, 3800, 1013, 4432, 1013, 2449, 3289, 1997, 1996, 4031, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2003, 2393, 3223, 2000, 3188, 25090, 4371, 8518, 5834, 2006, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2040, 2515, 1996, 3188, 25090, 9276, 1997, 8518, 2241, 2006, 5918, 1013, 17060, 2015, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2040, 2515, 1996, 3188, 25090, 9276, 1997, 8518, 2241, 2006, 5918, 1013, 17060, 2015, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3446, 4824, 1997, 1996, 3800, 1013, 4432, 1013, 2449, 3289, 1997, 1996, 4031, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2247, 2007, 1996, 2136, 2013, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102, 0, 0, 0], [101, 2007, 3183, 2024, 9095, 18856, 8486, 10803, 2015, 6148, 2098, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3446, 4824, 1997, 1996, 3800, 1013, 4432, 1013, 2449, 3289, 1997, 1996, 4031, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 12151, 16680, 1999, 5918, 1998, 6224, 18856, 8486, 10803, 2015, 2005, 1996, 2878, 11336, 1006, 2025, 2074, 2006, 1996, 2112, 2499, 1013, 5117, 2006, 1007, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2079, 9095, 16680, 3961, 1999, 24977, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2024, 1996, 5604, 4084, 8832, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2007, 3183, 2024, 9095, 18856, 8486, 10803, 2015, 6148, 2098, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2007, 3183, 2024, 9095, 18856, 8486, 10803, 2015, 6148, 2098, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2247, 2007, 1996, 2136, 2013, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102, 0, 0, 0], [101, 2129, 4703, 4194, 1999, 23138, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 24977, 2031, 9751, 2042, 2589, 2011, 2969, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 12151, 16680, 1999, 5918, 1998, 6224, 18856, 8486, 10803, 2015, 2005, 1996, 2878, 11336, 1006, 2025, 2074, 2006, 1996, 2112, 2499, 1013, 5117, 2006, 1007, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 4194, 1999, 23138, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 3446, 1996, 4824, 1997, 1996, 3800, 1013, 3289, 1997, 1996, 14184, 2499, 2006, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2247, 2007, 1996, 2136, 2013, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102, 0, 0, 0], [101, 2042, 3223, 2000, 2191, 2640, 1013, 7375, 3119, 1011, 12446, 2241, 2006, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2011, 2969, 1006, 2203, 2000, 2203, 1007, 2007, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102], [101, 2042, 3223, 2000, 2191, 2640, 1013, 7375, 3119, 1011, 12446, 2241, 2006, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 4194, 1999, 23138, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2024, 1996, 5604, 4084, 8832, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2011, 2969, 1006, 2203, 2000, 2203, 1007, 2007, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2247, 2007, 1996, 2136, 2013, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102, 0, 0, 0], [101, 3446, 1996, 4824, 1997, 1996, 3800, 1013, 3289, 1997, 1996, 14184, 2499, 2006, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 12151, 16680, 1999, 5918, 1998, 6224, 18856, 8486, 10803, 2015, 2005, 1996, 2878, 11336, 1006, 2025, 2074, 2006, 1996, 2112, 2499, 1013, 5117, 2006, 1007, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2079, 9095, 16680, 3961, 1999, 24977, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 2092, 2020, 2640, 1013, 7375, 3119, 1011, 12446, 2081, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2007, 3183, 2024, 9095, 18856, 8486, 10803, 2015, 6148, 2098, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2042, 3223, 2000, 2191, 2640, 1013, 7375, 3119, 1011, 12446, 2241, 2006, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2024, 1996, 5604, 4084, 8832, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2003, 2393, 3223, 2000, 3188, 25090, 4371, 8518, 5834, 2006, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 4194, 1999, 23138, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 4703, 2079, 9095, 16680, 3961, 1999, 24977, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 2092, 2020, 2640, 1013, 7375, 3119, 1011, 12446, 2081, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2005, 2054, 2946, 14184, 2031, 9095, 18856, 8486, 10803, 2015, 2042, 6148, 2098, 2011, 2969, 1006, 2203, 2000, 2203, 1007, 2007, 1996, 2364, 8406, 14528, 1029, 1006, 2946, 1999, 2193, 1997, 2420, 1007, 1012, 102], [101, 2003, 2393, 3223, 2000, 3188, 25090, 4371, 8518, 5834, 2006, 5918, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2040, 2515, 1996, 3188, 25090, 9276, 1997, 8518, 2241, 2006, 5918, 1013, 17060, 2015, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2129, 2092, 2020, 2640, 1013, 7375, 3119, 1011, 12446, 2081, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 2007, 3183, 2024, 9095, 18856, 8486, 10803, 2015, 6148, 2098, 1029, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
t = 1
for record in x:
    print(t)
    print(len(record))
    print(record)
    t = t + 1
