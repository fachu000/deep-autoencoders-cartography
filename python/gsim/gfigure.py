import sys
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['b', 'r', 'g', 'm', 'k', 'c', 'r', 'g', 'g', 'g', 'g'],
                                             linestyle=['-', '-', '-.', '--', ':', '--', '--',  '--', '-.', ':', '--'],
                                             marker=['d', '>', 's', '*', 'o', 'v', ' ', ' ', ' ', ' ', ' '])

from IPython.core.debugger import set_trace
import copy
import numpy as np
"""

The easiest way to learn how to use this module is to run the examples
at the end.

"""


class Curve:
    def __init__(self, xaxis=None, yaxis=[], style=None, legend_str=""):
        """
        
        xaxis : list of a numeric type or None. In the former case, its length 
            equal the length of yaxis.

        yaxis : list of a numeric type. 

        style : str used as argument to plt.plot()
        
        """

        # Input check
        if type(yaxis) != list:
            set_trace()
            raise TypeError("`yaxis` must be a list of numeric entries")
        if type(xaxis) == list:
            assert len(xaxis) == len(yaxis)
        elif xaxis is not None:
            raise TypeError(
                "`xaxis` must be a list of numeric entries or None")
        if (style is not None) and (type(style) != str):
            raise TypeError("`style` must be of type str or None")
        if type(legend_str) != str:
            raise TypeError("`legend_str` must be of type str")

        # Save
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.style = style
        self.legend_str = legend_str

    def plot(self):

        if type(self.xaxis) == list and len(self.xaxis):
            if self.style:
                plt.plot(self.xaxis,
                         self.yaxis,
                         self.style,
                         label=self.legend_str)
            else:
                plt.plot(self.xaxis, self.yaxis, label=self.legend_str)
        else:
            if self.style:
                plt.plot(self.yaxis, self.style, label=self.legend_str)
            else:
                plt.plot(self.yaxis, label=self.legend_str)

    def legend_is_empty(l_curves):

        for curve in l_curves:
            if curve.legend_str != "":
                return False
        return True

    #     b_empty_legend = True
    #     for curve in l_curves:
    #         if curve.legend_str != "":
    #             b_empty_legend = False
    #             break

    #     if b_empty_legend:
    #         return tuple([])
    #     else:
    #         return tuple([curve.legend_str for curve in l_curves])


class Subplot:
    def __init__(self, title="", xlabel="", ylabel="", grid=True, xlim=None, ylim=None, **kwargs):
        """
        For a description of the arguments, see GFigure.
        
        """

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.grid = grid
        self.xlim = xlim
        self.ylim = ylim

        #    self.l_curves = Subplot._l_curve_from_input_args(xaxis, yaxis, styles,                                                      legend)
        self.l_curves = []
        self.add_curve(**kwargs)

    def is_empty(self):

        return not any([self.title, self.xlabel, self.ylabel, self.l_curves])

    def update_properties(self, **kwargs):

        if "title" in kwargs:
            self.title = kwargs["title"]
        if "xlabel" in kwargs:
            self.xlabel = kwargs["xlabel"]
        if "ylabel" in kwargs:
            self.ylabel = kwargs["ylabel"]

    def add_curve(self, xaxis=[], yaxis=[], styles=[], legend=tuple()):
        """

        """

        self.l_curves += Subplot._l_curve_from_input_args(
            xaxis, yaxis, styles, legend)

    def _l_curve_from_input_args(xaxis, yaxis, styles, legend):

        # Process the subplot input.  Each entry of l_xaxis or l_yaxis is
        # a list of a numerical type. Both lists will have the same length.
        l_xaxis, l_yaxis = Subplot._list_from_axis_arguments(xaxis, yaxis)
        l_style = Subplot._list_from_style_argument(styles)

        # Process style input.
        if len(l_style) == 0:
            l_style = [None] * len(l_xaxis)
        elif len(l_style) == 1:
            l_style = l_style * len(l_xaxis)
        else:
            if len(l_style) < len(l_xaxis):
                set_trace()
            assert len(l_style) >= len(l_xaxis), "The length of `style` must be"\
                " either 1 or no less than the number of curves"

        # Process the legend
        assert ((type(legend) == tuple) or (type(legend) == list)
                or (type(legend) == str))
        if type(legend) == str:
            legend = [legend] * len(l_xaxis)
        else:  # legend is tuple or list
            if len(legend) == 0:
                legend = [""] * len(l_xaxis)
            else:
                assert type(
                    legend[0]
                ) == str, "`legend` must be an str, list of str, or tuple of str"
                assert (len(legend) == len(l_xaxis)
                        ), "len(legend) must equal 0 or the number of curves"

        b_debug = True
        if b_debug:
            conditions = [
                len(l_xaxis) == len(l_yaxis),
                len(l_xaxis) == len(l_style),
                type(l_xaxis) == list,
                type(l_yaxis) == list,
                type(l_style) == list,
                (len(l_xaxis) == 0) or (type(l_xaxis[0]) == list)
                or (l_xaxis[0] is None),
                (len(l_yaxis) == 0) or (type(l_yaxis[0]) == list)
                or (l_yaxis[0] is None),
                (len(l_style) == 0) or (type(l_style[0]) == str)
                or (l_style[0] is None),
            ]
            if not np.all(conditions):
                print(conditions)
                set_trace()

        # Construct Curve objects
        l_curve = []
        for xax, yax, stl, leg in zip(l_xaxis, l_yaxis,
                                      l_style[0:len(l_xaxis)], legend):
            l_curve.append(
                Curve(xaxis=xax, yaxis=yax, style=stl, legend_str=leg))
        return l_curve

    def _list_from_style_argument(style_arg):
        """
        Returns a list of str. 
        """
        err_msg = "Style argument must be an str "\
            "or list of str"
        if type(style_arg) == str:
            return [style_arg]
        elif type(style_arg) == list:
            for entry in style_arg:
                if type(entry) != str:
                    raise TypeError(err_msg)
            return copy.copy(style_arg)
        else:
            raise TypeError(err_msg)

    def is_number(num):
        #return isinstance(num, (int, float, complex, bool))
        # From https://stackoverflow.com/questions/500328/identifying-numeric-and-array-types-in-numpy
        attrs = ['__add__', '__sub__', '__mul__', '__truediv__', '__pow__']
        return all(hasattr(num, attr) for attr in attrs)

    def _list_from_axis_arguments(xaxis_arg, yaxis_arg):
        """Processes subplot arguments and returns two lists of the same length
        whose elements can be either None or lists of a numerical
        type. None means "use the default x-axis for this curve".

        """
        def unify_format(axis):
            def ndarray_to_list(arr):
                assert (type(arr) == np.ndarray)
                if arr.ndim == 1:
                    return [list(arr)]
                elif arr.ndim == 2:
                    return [[arr[row, col] for col in range(0, arr.shape[1])]
                            for row in range(0, arr.shape[0])]
                else:
                    raise TypeError

            if (type(axis) == np.ndarray):
                return ndarray_to_list(axis)
            elif (type(axis) == list):
                if len(axis) == 0:
                    return []
                if Subplot.is_number(axis[0]):
                    return [copy.copy(axis)]
                else:
                    out_list = []
                    for entry in axis:
                        if type(entry) == np.ndarray:
                            if entry.ndim == 1:
                                out_list.append(copy.copy(entry))
                            else:
                                raise Exception(
                                    "Arrays inside the list must be 1D in the current implementation"
                                )
                        elif type(entry) == list:
                            if len(entry) == 0:
                                out_list.append([])
                            elif Subplot.is_number(entry[0]):
                                out_list.append(copy.copy(entry))
                            else:
                                raise TypeError
                    return out_list
            elif axis is None:
                return [None]
            else:
                raise TypeError

        # Construct two lists of possibly different lengths.
        l_xaxis = unify_format(xaxis_arg)
        l_yaxis = unify_format(yaxis_arg)

        # Expand lists if needed to have the same length
        str_message = "Number of curves in the xaxis must be"\
            " 1 or equal to the number of curves in the y axis"
        if len(l_xaxis) > 0 and len(l_yaxis) != len(l_xaxis):
            raise Exception(str_message)
        if len(l_xaxis) == 0 and len(l_yaxis) > 0:
            l_xaxis = [None]
        if len(l_yaxis) > 1:
            if len(l_xaxis) == 1:
                l_xaxis = l_xaxis * len(l_yaxis)
            if len(l_xaxis) != len(l_yaxis):
                raise Exception(str_message)
        elif len(l_yaxis) == 1:
            if len(l_xaxis) != 1:
                raise Exception(str_message)

        return l_xaxis, l_yaxis

    def plot(self):

        for curve in self.l_curves:
            curve.plot()


#        plt.legend(Curve.list_to_legend(self.l_curves))
        if not Curve.legend_is_empty(self.l_curves):
            plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if self.title:
            plt.title(self.title)

        if "grid" in dir(self): # backwards compatibility
            plt.grid(self.grid)

        if "xlim" in dir(self): # backwards compatibility
            if self.xlim:
                plt.xlim(self.xlim)

        if "ylim" in dir(self): # backwards compatibility
            if self.ylim:
                plt.ylim(self.ylim)
            
        return


class GFigure:
    def __init__(self,
                 *args,
                 figsize=None,
                 ind_active_subplot=0,
                 num_subplot_rows=None,
                 num_subplot_columns=1,
                 **kwargs):
        """Arguments of mutable types are (deep) copied so they can be
        modified by the user after constructing the GFigure object
        without altering the figure.
        
        SUBPLOT ARGUMENTS:

        The first set of arguments allow the user to create a subplot when 
        creating the GFigure object.

        title : str 

        xlabel : str

        ylabel : str

        grid : bool

        xlim : tuple

        ylim : tuple

        CURVE ARGUMENTS:

        xaxis and yaxis:
            (a) To specify only one curve:
                - `yaxis` can be a list of a numeric type or 1D np.ndarray
                - `xaxis` can be None, a list of a numeric type, or a 1D 
                np.array of the same length as `yaxis`.
            (b) To specify one or more curves:
                - `yaxis` can be:
                    -> a list of the types specified in (a)
                    -> M x N np.ndarray. Each row corresponds to a curve.
                - `xaxis` can be either as in (a), so all curves share the same 
                X-axis points, or
                    -> a list of the types specified in (a)
                    -> Mx x N np.ndarray. Each row corresponds to a curve. Mx 
                    must be either M or 1. 

        styles: specifies the style argument to plot, as in MATLAB. Possibilities:
            - str : this style is applied to all curves specified by 
             `xaxis` and `yaxis` will 
            - list of str : then style[n] is applied to the n-th curve. Its length
              must be at least the number of curves.

        legend : str, tuple of str, or list of str. If the str begins with "_", then
            that curve is not included in the legend.


        ARGUMENTS FOR SPECIFYING HOW TO SUBPLOT:

        
       `ind_active_subplot`: The index of the subplot that is created and
            where new curves will be added until a different value for
            the property of GFigure with the same name is specified. A
            value of 0 refers to the first subplot.

        `num_subplot_rows` and `num_subplot_columns` determine the
            number of subplots in each column and row respectively. If
            None, their value is determined by the value of the other
            of these parameters and the number of specified
            subplots. If the number of specified subplots does not
            equal num_subplot_columns*num_subplot_rows, then the value
            of num_subplot_columns is determined from the number of
            subplots and num_subplot_rows.

            The values of the properties of GFigure with the same name
            can be specified subsequently.

        """

        new_subplot = Subplot(*args, **kwargs)
        #        set_trace()
        self.ind_active_subplot = ind_active_subplot

        # List of axes to create subplots
        # self.l_subplots = [None] * (self.ind_active_subplot + 1)
        # self.l_subplots[self.ind_active_subplot] = new_subplot
        if not new_subplot.is_empty():
            # List of axes to create subplots
            self.l_subplots = [None]*(self.ind_active_subplot+1)
            self.l_subplots[self.ind_active_subplot] = new_subplot
        else:
            self.l_subplots = []

        self.num_subplot_rows = num_subplot_rows
        self.num_subplot_columns = num_subplot_columns
        self.figsize = figsize

    def add_curve(self, *args, ind_active_subplot=None, **kwargs):
        """
           Similar arguments to __init__ above.

        
        """

        # Modify ind_active_subplot only if provided
        if ind_active_subplot is not None:
            self.ind_active_subplot = ind_active_subplot

        self.select_subplot(self.ind_active_subplot, **kwargs)
        self.l_subplots[self.ind_active_subplot].add_curve(*args, **kwargs)

    def next_subplot(self, **kwargs):
        # Creates a new subplot at the end of the list of axes. One can
        # specify subplot parameters; see GFigure.
        self.ind_active_subplot = len(self.l_subplots)
        if kwargs:
            self.l_subplots.append(Subplot(**kwargs))

    def select_subplot(self, ind_subplot, **kwargs):
        # Creates the `ind_subplot`-th subplot if it does not exist and
        # selects it. Subplot keyword parameters can also be provided;
        # see GFigure.

        self.ind_active_subplot = ind_subplot

        # Complete the list l_subplots if index self.ind_active_subplot does
        # not exist.
        if ind_subplot >= len(self.l_subplots):
            self.l_subplots += [None] * (self.ind_active_subplot -
                                         len(self.l_subplots) + 1)

        # Create if it does not exist
        if self.l_subplots[self.ind_active_subplot] is None:
            self.l_subplots[self.ind_active_subplot] = Subplot(**kwargs)
        else:
            self.l_subplots[self.ind_active_subplot].update_properties(
                **kwargs)

    def plot(self):

        # backwards compatibility
        if "figsize" not in dir(self):
            figsize = None
        else:
            figsize = self.figsize

        F = plt.figure(figsize=figsize)
        #plt.tight_layout()
        
        num_axes = len(self.l_subplots)
        if self.num_subplot_rows is not None:
            self.num_subplot_columns = int(
                np.ceil(num_axes / self.num_subplot_rows))
        else:  # self.num_subplot_rows is None
            if self.num_subplot_columns is None:
                # Both are None. Just arrange thhe plots as a column
                self.num_subplot_columns = 1
                self.num_subplot_rows = num_axes
            else:
                self.num_subplot_rows = int(
                    np.ceil(num_axes / self.num_subplot_columns))

        for index, subplot in enumerate(self.l_subplots):
            plt.subplot(self.num_subplot_rows, self.num_subplot_columns,
                        index + 1)
            if self.l_subplots[index] is not None:
                self.l_subplots[index].plot()


        
        return F
        


def example_figures(ind_example):

    v_x = np.linspace(0, 10, 20)
    v_y1 = v_x**2 - v_x + 3
    v_y2 = v_x**2 + v_x + 3
    v_y3 = v_x**2 - 2 * v_x - 10

    if ind_example == 1:
        # Example with a single curve, single subplot
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")
    elif ind_example == 2:
        # Example with three curves on one subplot
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")
        G.add_curve(xaxis=v_x, yaxis=v_y2, legend="P2")
        G.add_curve(xaxis=v_x, yaxis=v_y3, legend="P3")
    elif ind_example == 3:
        # Typical scheme where a simulation function produces each
        # curve.
        def my_simulation():
            coef = np.random.random()
            v_y_new = coef*v_y1
            G.add_curve(xaxis=v_x, yaxis=v_y_new, legend="coef = %.2f"%coef)

        """ One can specify the axis labels and title when the figure is
        created."""        
        G = GFigure(xlabel="x", ylabel="f(x)", title="Parabola") 
        for ind in range(0, 6):
            my_simulation()
    elif ind_example == 4:
        # Example with two subplots
        G = GFigure(xaxis=v_x,
                    yaxis=v_y1,
                    xlabel="x",
                    ylabel="f(x)",
                    title="Parabolas",
                    legend="P1")
        G.add_curve(xaxis=v_x, yaxis=v_y2, legend="P2")
        G.next_subplot(xlabel="x")
        G.add_curve(
            xaxis=v_x,
            yaxis=v_y3,
            legend="P3",
        )
    elif ind_example == 5:
        # Example with a large multiplot
        G = GFigure(num_subplot_rows=4)
        for ind in range(0, 12):
            G.select_subplot(ind, xlabel="x", ylabel="f(x)", title="Parabolas")
            G.add_curve(xaxis=v_x, yaxis=v_y1, legend="P1", styles="r")
    elif ind_example == 6:
        # Typical scheme where a simulation function produces each subplot
        def my_simulation():
            G.next_subplot(xlabel="x", ylabel="f(x)", title="Parabola")
            G.add_curve(xaxis=v_x, yaxis=v_y1, legend="P1", styles="r")

        """ Important not to specify axis labels or the title in the next line
        because that would create an initial subplot without curves
        and, therefore, function `next_subplot` will move to the
        second subplot of the figure the first time `my_simulation` is
        executed."""
        
        G = GFigure(num_subplot_rows=3) 
        for ind in range(0, 6):
            my_simulation()

    G.plot()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("""Usage from command line: 
$ python3 gfigure.py <example_index>
            
where <example_index> is an integer. See function `example_figures`.""")
    else:
        example_figures(int(sys.argv[1]))
