import os
import tensorwatch as tw
import argparse
from itertools import cycle
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.layouts import column
import numpy as np

parser = argparse.ArgumentParser(
    description='Probe experiments done with convNet.pytorch using tensorwatch')
parser.add_argument('experiments', metavar='N', type=str, nargs='+',
                    help='pathes to multiple experiments')
parser.add_argument('--legend', type=str, nargs='+',
                    help='legend to multiple experiments')
parser.add_argument('--x-axis', type=str, default='epochs',
                    help='x axis to experiments')
parser.add_argument('--plot', type=str, nargs='+', default=['step', 'data', 'loss', 'prec1', 'prec5', 'grad'],
                    help='values to plot')
parser.add_argument('--colors', type=str, nargs='+',
                    default=['red', 'green', 'blue', 'orange',
                             'black', 'purple', 'brown'],
                    help='color for each experiment')
parser.add_argument('--all', action='store_true', default=False,
                    help='show data for every step (not only epochs)')

PUBLISH_MODE = False
if PUBLISH_MODE:
    from bokeh.plotting import figure as bk_figure
    text_font = 'helvetica'
    title_text_font_size = '14pt'
    x_axis_label_text_font_size = '24pt'
    y_axis_label_text_font_size = '24pt'
    legend_text_font_size = '20pt'
    major_label_text_font_size = '24pt'

    def figure(*kargs, **kwargs):
        fig = bk_figure(*kargs, **kwargs)
        fig.title.text_font_size = title_text_font_size
        fig.xaxis.axis_label_text_font_size = x_axis_label_text_font_size
        fig.yaxis.axis_label_text_font_size = y_axis_label_text_font_size
        fig.xaxis.major_label_text_font_size = major_label_text_font_size
        fig.yaxis.major_label_text_font_size = major_label_text_font_size
        return fig
else:
    from bokeh.plotting import figure


def multi_line_opts(num,
                    dashes=None,
                    colors=['red', 'green', 'blue', 'orange',
                            'black', 'purple', 'brown'],
                    **defaults):
    defaults.setdefault('line_width', 2)
    options = []
    colors = cycle(colors)
    if dashes is not None:
        dashes = cycle(dashes)

    for _ in range(num):
        if dashes is not None:
            defaults.update(line_dash=next(dashes))
        options.append({'line_color': next(colors), **defaults})
    return options
    # line_opts = dict(line_width=line_width,
    #                  line_color=next(colors), line_dash=[6, 3], legend=name)


def reduce_multiple(result):
    x_values = [x for x, _ in result]
    x_unq = sorted(set(x_values), key=x_values.index)
    if len(x_unq) != len(x_values):
        new_result = []
        for curr_x in x_unq:
            y_values = [y for x, y in result if curr_x == x]
            aggr_y = sum(y_values) / len(y_values)
            new_result.append((curr_x, aggr_y))
        result = new_result
    return result


def plot_comparison(experiments,
                    line_options=None,
                    title=None,
                    x_axis_label=None,
                    y_axis_label=None,
                    x_axis_type='linear',
                    y_axis_type='linear',
                    x_range=None,
                    y_range=None,
                    width=800,
                    height=400,
                    tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save',):
    line_options = line_options or multi_line_opts(len(experiments))
    if len(line_options) < len(experiments):
        line_options += multi_line_opts(len(experiments) -
                                        len(line_options))
    fig = figure(title=title, tools=tools,
                 width=width, height=height,
                 x_axis_label=x_axis_label,
                 y_axis_label=y_axis_label,
                 x_axis_type=x_axis_type,
                 y_axis_type=y_axis_type,
                 x_range=x_range,
                 y_range=y_range)
    plotted = False
    for i, (name, result) in enumerate(experiments.items()):
        if result is None or len(result) == 0:
            continue
        result_x, result_y = zip(*result)
        fig.line(result_x, result_y, legend=name, **line_options[i])
        plotted = True
    if plotted:
        fig.legend.click_policy = "hide"
    else:
        fig = None
    return fig


def dump_stream(client, stream_name):
    stream = client.open_stream(name=stream_name)
    output = [getattr(entry, 'value') for entry in stream.read_all()]
    return reduce_multiple(output)


def reduce_ranges(results, x_values):
    # assumes x_values are sorted unique values
    x_values = np.array(x_values)
    new_results = []
    for x, y in results:
        idx = x_values.searchsorted(x)
        if idx < len(x_values):
            x = x_values[idx]
        new_results.append((x, y))
    return reduce_multiple(new_results)

    # line_dash=[6, 3]


def annotate_indices(results, x_values):
    x_values = np.array(x_values)
    new_results = []
    for x, y in results:
        idx = x_values.searchsorted(x)
        new_results.append((idx, y))
    return reduce_multiple(new_results)


if __name__ == '__main__':
    args = parser.parse_args()

    if len(args.experiments) > 1:
        exit()

    exp = args.experiments[0]
    client = tw.WatcherClient(filename=os.path.join(exp, 'tensorwatch.log'))
    meter_names = ['loss', 'prec1', 'prec5', 'grad', 'step', 'data']
    additional = ['lr']

    defaults_streams = ['train_' + n for n in meter_names] +\
        ['eval_' + n for n in meter_names]

    epoch_steps = [x for x, _ in dump_stream(client, 'epoch')]

    if args.x_axis == 'epochs':
        args.all = False

    figures = []
    for name in meter_names:
        train_metrics = dump_stream(client, 'train_%s' % name)
        val_metrics = dump_stream(client, 'eval_%s' % name)
        if not args.all:
            train_metrics = reduce_ranges(train_metrics, epoch_steps)
            val_metrics = reduce_ranges(val_metrics, epoch_steps)
        if args.x_axis == 'epochs':
            train_metrics = annotate_indices(train_metrics, epoch_steps)
            val_metrics = annotate_indices(val_metrics, epoch_steps)
        metrics = {'Training %s' % name: train_metrics,
                   'Validation %s' % name: val_metrics}
        fig = plot_comparison(metrics, title='%s values' % name, line_options=multi_line_opts(len(metrics)),
                              x_axis_label=args.x_axis, y_axis_label=name)
        if fig is not None:
            figures.append(fig)

    for name in additional:
        metrics = dump_stream(client, name)
        if not args.all:
            metrics = reduce_ranges(metrics, epoch_steps)
        if args.x_axis == 'epochs':
            metrics = annotate_indices(metrics, epoch_steps)

        fig = plot_comparison({name: metrics}, title=name, line_options=multi_line_opts(1),
                              x_axis_label=args.x_axis, y_axis_label=name)
        if fig is not None:
            figures.append(fig)
    plots = column(*figures)
    save(plots)
    # values = {name: dump_stream(client, name) for name in defaults_streams}
