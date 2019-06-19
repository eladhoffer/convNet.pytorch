import os
import tensorwatch as tw
import argparse
from itertools import cycle
from collections import OrderedDict
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.layouts import column
from bokeh.plotting import figure as bk_figure
import numpy as np

parser = argparse.ArgumentParser(
    description='Probe experiments done with convNet.pytorch using tensorwatch')
parser.add_argument('experiments', metavar='N', type=str, nargs='+',
                    help='pathes to multiple experiments')
parser.add_argument('--legend', type=str, nargs='+',
                    help='legend to multiple experiments')
parser.add_argument('--metrics', type=str, nargs='+', default=['loss', 'prec1', 'prec5', 'grad'],
                    help='metrics (train/val) to extract')
parser.add_argument('--additional', type=str, nargs='+', default=['lr'],
                    help='additional metrics (standalone) to extract')
parser.add_argument('--x-axis', type=str, default='steps',
                    help='x axis to experiments')
parser.add_argument('--colors', type=str, nargs='+',
                    default=['green', 'red', 'blue', 'orange',
                             'black', 'purple', 'brown'],
                    help='color for each experiment')
parser.add_argument('--all', action='store_true', default=False,
                    help='show data for every step (not only epochs)')
parser.add_argument('--paper', action='store_true', default=False,
                    help='publish ready plots')


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
                    figure=None,
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
                    legend_text_font_size=None,
                    tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save',
                    figure_fn=bk_figure):
    line_options = line_options or multi_line_opts(len(experiments))
    if len(line_options) < len(experiments):
        line_options += multi_line_opts(len(experiments) -
                                        len(line_options))
    if figure is None:
        figure = figure_fn(title=title, tools=tools,
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
        figure.line(result_x, result_y, legend=name, **line_options[i])
        plotted = True
    if plotted:
        figure.legend.click_policy = "hide"
        if legend_text_font_size is not None:
            figure.legend.label_text_font_size = legend_text_font_size
    else:
        figure = None
    return figure


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


def annotate_indices(results, x_values, interpolate=False):
    x_values = np.array(x_values)
    new_results = []
    for x, y in results:
        idx = x_values.searchsorted(x)
        if interpolate and idx < len(x_values) - 1:
            length = float(x_values[idx+1] - x_values[idx])
            idx = float(idx) + (x - x_values[idx]) / length
        new_results.append((idx, y))
    return reduce_multiple(new_results)


if __name__ == '__main__':
    args = parser.parse_args()
    metrics = args.metrics
    additional = args.additional

    if args.paper:
        def figure_fn(*kargs, **kwargs):
            fig = bk_figure(*kargs, **kwargs)
            fig.title.text_font_size = '14pt'
            fig.xaxis.axis_label_text_font_size = '24pt'
            fig.yaxis.axis_label_text_font_size = '24pt'
            fig.xaxis.major_label_text_font_size = '24pt'
            fig.yaxis.major_label_text_font_size = '24pt'
            return fig
        line_width = 3
        plot_defaults = dict(figure_fn=figure_fn, legend_text_font_size='20pt',
                             width=1000, height=500)
    else:
        line_width = 2
        plot_defaults = dict(figure_fn=bk_figure,
                             width=800, height=400)

    defaults_streams = ['train_' + n for n in metrics] +\
        ['eval_' + n for n in metrics]

    if len(args.experiments) > 1:
        exp_names = args.legend or [exp.replace(
            './results', '') for exp in args.experiments]
    else:
        exp_names = ['']

    results = OrderedDict()

    for exp_name, exp in zip(exp_names, args.experiments):
        client = tw.WatcherClient(
            filename=os.path.join(exp, 'tensorwatch.log'))
        epoch_steps = [x for x, _ in dump_stream(client, 'epoch')]
        prefix = exp_name + ' - ' if exp_name != '' else ''

        for metric_name in metrics:
            results.setdefault(metric_name, {})
            train_metrics = dump_stream(client, 'train_%s' % metric_name)
            val_metrics = dump_stream(client, 'eval_%s' % metric_name)
            if not args.all:
                train_metrics = reduce_ranges(train_metrics, epoch_steps)
                val_metrics = reduce_ranges(val_metrics, epoch_steps)
            if args.x_axis == 'epochs':
                train_metrics = annotate_indices(
                    train_metrics, epoch_steps, interpolate=args.all)
                val_metrics = annotate_indices(
                    val_metrics, epoch_steps, interpolate=args.all)
            results[metric_name].update({'%svalidation %s' % (prefix, metric_name): val_metrics,
                                         '%straining %s' % (prefix, metric_name): train_metrics})

        for name in additional:
            results.setdefault(name, {})
            metric = dump_stream(client, name)
            if not args.all:
                metric = reduce_ranges(metric, epoch_steps)
            if args.x_axis == 'epochs':
                metric = annotate_indices(metric, epoch_steps)
            results[name].update({'%s%s' % (prefix, name): metric})

    line_options = OrderedDict()
    if len(args.experiments) == 1:
        # Training / val in different color
        line_options.update({metric_name: multi_line_opts(2, line_width=line_width)
                             for metric_name in metrics})
        line_options.update({name: multi_line_opts(1, line_width=line_width)
                             for name in additional})
    else:
        # Training / val in different line style, experiments in different color
        line_options_exp = multi_line_opts(
            len(args.experiments), line_width=line_width)
        for i in range(len(args.experiments)):
            train_ops = {'line_dash': [6, 3], **line_options_exp[i]}
            val_ops = line_options_exp[i]
            for metric_name in metrics:
                line_options.setdefault(metric_name, [])
                line_options[metric_name] += [val_ops, train_ops]
            for name in additional:
                line_options.setdefault(name, [])
                line_options[name] += [line_options_exp[i]]

    figures = []
    for name, result in results.items():
        fig = plot_comparison(result, title=name, line_options=line_options[name],
                              x_axis_label=args.x_axis, y_axis_label=name, **plot_defaults)
        if fig is not None:
            figures.append(fig)

    plots = column(*figures)
    save(plots)
