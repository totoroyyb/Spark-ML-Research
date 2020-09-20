# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:39:30 2020

@author: yibo yan
"""

import matplotlib.pyplot as plt
import glob
import argparse
import os
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def extract_data(variable, path):
    path = os.path.join(path, variable)
    files = glob.glob(path + "-*")
    
    if not files:
        print("Cannot find any file.")
        return
    
    params = list(map(lambda x: float(x[x.rfind("-") + 1:]), files))
    values = []
    
    for file in files:
        with open(file, "r") as f:
            first_line = f.readline()
            values.append(float(first_line[first_line.find(":") + 1:]))
            
    return params, values
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", default="hidden_size", type=str,
                        help="The variable which is used to generate plot.")
    parser.add_argument("--sub_path", default=True, type=bool,
                        help="The default path is ./output")
    parser.add_argument("--save", default=True, type=bool,
                        help="Whether save the plot or not.")
    parser.add_argument("--type", default="line", type=str,
                        help="The type of chart, can be line or bar.")
    
    args = parser.parse_args()
    
    data_path = "./output"
    
    if args.sub_path:
        data_path = os.path.join(data_path, args.variable)
    
    params, values = extract_data(args.variable, data_path)
    
    if params == None or values == None or not params or not values:
        print("Operation failed.")
        exit(1)
    
    params, values = zip(*sorted(zip(params, values)))
    
    plt.style.use('ggplot')
    
    plt.title("Validation Accuracy vs {}".format(args.variable))
    plt.ylabel("Validation Accuracy")
    plt.xlabel(args.variable)
    
    if args.type == "bar":
        fig, ax = plt.subplots()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        
        x_pos = [i for i, _ in enumerate(params)]
        
        upper = max(values) + 0.01
        lower = min(values) - 0.01
        stride = 0.005
        plt.ylim(lower, upper)
        plt.yticks(np.arange(lower, upper, step=stride))
        
        plt.bar(x_pos, values, color='green')
        plt.xticks(x_pos, params, rotation=90)
    elif args.type == "line":
        plt.plot(params, values)
    
    if args.save:
        plot_path = os.path.join(data_path, "{}-{}.png".format(args.variable, args.type))
        plt.savefig(plot_path, bbox_inches='tight',dpi=100)
        print("Plot has been saved to {}.".format(plot_path))
    
    plt.show()
    
    
