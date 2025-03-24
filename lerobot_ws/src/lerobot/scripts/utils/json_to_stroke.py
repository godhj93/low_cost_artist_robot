import json
import copy
import pandas as pd

def refine_stroke_and_to_csv(refined_filename, filename=None, stroke_list=None):
    if filename is not None:
        with open(filename) as f:
            stroke_list = json.load(f)

    refined_strokes = []    
    for strokes in stroke_list:
        refined_diagram = []
        for lines in strokes:
            for line in lines:
                refined_diagram.append([line[0], line[1], 0])
        
        refined_diagram = copy.deepcopy(refined_diagram[:1]) + refined_diagram + copy.deepcopy(refined_diagram[-1:])
        refined_diagram[0][-1] = 1
        refined_diagram[-1][-1] = 1
        
        refined_strokes += copy.deepcopy(refined_diagram)

    with open(f'{refined_filename}.csv', 'w') as f:
        f.write("x,y,z\n")
        for i in refined_strokes:
            f.write(f"{i[0]},{i[1]},{i[2]}\n")
        f.close()
        
        refined_strokes_df = pd.DataFrame(refined_strokes, columns=['x', 'y', 'z'], index=None)
        refined_strokes_df.to_csv(f'{refined_filename}.csv', index=False)
        
    return refined_strokes_df
    