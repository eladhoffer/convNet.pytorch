import argparse
from itertools import cycle
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
import pandas as pd

parser = argparse.ArgumentParser(
    description='Compare experiments done with convNet.pytorch')
parser.add_argument('experiments', metavar='N', type=str, nargs='+',
                    help='pathes to multiple experiments')
parser.add_argument('--legend', type=str, nargs='+',
                    help='legend to multiple experiments')
parser.add_argument('--x_axis', type=str, default='epoch',
                    help='x axis to experiments')
parser.add_argument('--compare', type=str, nargs='+', default=['training error1', 'validation error1'],
                    help='values to compare')
parser.add_argument('--colors', type=str, nargs='+',
                    default=['red', 'green', 'blue', 'orange',
                             'black', 'purple', 'brown'],
                    help='color for each experiment')


def main():
    args = parser.parse_args()
    title = 'comparison: ' + ','.join(args.experiments)
    x_axis_type = 'linear'
    y_axis_type = 'linear'
    width = 800
    height = 400
    line_width = 2
    tools = 'pan,box_zoom,wheel_zoom,box_select,hover,reset,save'
    results = {}
    for i, exp in enumerate(args.experiments):
        if args.legend is not None and len(args.legend) > i:
            name = args.legend[i]
        else:
            name = exp
        filename = exp + '/results.csv'
        results[name] = pd.read_csv(filename, index_col=None)
    figures = []
    for comp in args.compare:
        fig = figure(title=comp, tools=tools,
                     width=width, height=height,
                     x_axis_label=args.x_axis,
                     y_axis_label=comp,
                     x_axis_type=x_axis_type,
                     y_axis_type=y_axis_type)
        colors = cycle(args.colors)
        for i, (name, result) in enumerate(results.items()):
            fig.line(result[args.x_axis], result[comp],
                     line_width=line_width,
                     line_color=next(colors), legend=name)
        fig.legend.click_policy = "hide"
        figures.append(fig)

    plots = column(*figures)
    show(plots)


if __name__ == '__main__':
    main()
