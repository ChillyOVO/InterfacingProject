import PDDTVisionToolBox as pd
import numpy as np

points = np.array([[100, 100, 0, 0],
                   [0, 0, 0, 0],
                   [100, 0, 100, 0]
                   ])

plane = pd.getPlaneFitting(points)
