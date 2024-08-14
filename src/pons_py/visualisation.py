import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib

NOTEBOOK = False
def set_matplotlib_backend():
    try:
        from IPython import get_ipython
        import matplotlib.pyplot as plt
        if 'IPKernelApp' in get_ipython().config:
            # We are running in a Jupyter Notebook
            print("Running in Jupyter Notebook. Setting backend to 'widget'.")
            matplotlib.use('widget')  # 'nbagg' is recommended for interactivity in Jupyter
            plt.ion()
            NOTEBOOK = True
        else:
            # We are running in a standard Python environment
            print("Running in a standard Python environment. Setting backend to 'default'.")
            NOTEBOOK = False
    except:
        # IPython is not available, so we are not in a notebook
        print("Not running in Jupyter Notebook. Setting backend to 'default'.")
        NOTEBOOK = False


# Set the backend before any plotting
set_matplotlib_backend()
print(f'Matplotlib Backend: {matplotlib.get_backend()}')

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import RadioButtons, Button, CheckButtons, RangeSlider
from .processing import Processing



class Visualisation:
    global NOTEBOOK
    def __init__(self, data=None):
        if data is not None:
            self.data = data
            print(f'Visualisation object created with {len(data)} rows of data and columms: {", ".join(data.columns)}.')
        self.POSSIBLE_VALUES = None
        self.VAR = None
        self.VAR_TYPE = None
    
    def align_layers(self,top, bottom, common, offset=10):
        """
        Vertically aligns the two layers so they are directly on top of each other. It does so by shifting the x values of one of the layers. Not intended to be called directly. Instead, this method is called by the multi_layer method in the Visualisation class.
        top: 2D numpy array of the coordinates (x,y) for the top layer. Can also be a pandas DataFrame.
        bottom: 2D numpy array of the coordinates (x,y) for the bottom. Can also be a pandas DataFrame.
        offset: The offset to be added to the y values of the top layer. This is influenced by the range of y-values. Default is 20.
        
        Returns:
        top: Aligned 2D numpy array of the coordinates (x,y) for the top layer.
        bottom: Aligned 2D numpy array of the coordinates (x,y) for the bottom layer.
        """
        
        common_top = common[top.columns.values].to_numpy()
        common_bottom = common[bottom.columns.values].to_numpy()
        
        if type(top) == pd.core.frame.DataFrame:
            top = top.dropna().to_numpy()
        if type(bottom) == pd.core.frame.DataFrame:
            bottom = bottom.dropna().to_numpy()

        # X Alignment
        top_mean = top[:, 0].mean()
        bottom_mean = bottom[:, 0].mean()
        diff = np.abs(top_mean - bottom_mean)

        # Shift the x values of one of the scatter plots to align their ranges
        if top_mean > bottom_mean:
            top[:, 0] = top[:, 0] - diff
            common_top[:, 0] = common_top[:, 0] - diff
        else:
            top[:, 0] =top[:, 0] + diff
            common_top[:, 0] = common_top[:, 0] + diff

        # Y Alignment
        if top[:, 1].min() > bottom[:, 1].max():
            offset = offset - (top[:, 1].min() - bottom[:, 1].max())
            top[:, 1] = top[:, 1] + offset
            common_top[:, 1] = common_top[:, 1] + offset
        else:
            offset = offset + (bottom[:, 1].min() - top[:, 1].max())
            top[:, 1] = top[:, 1] + offset
            common_top[:, 1] = common_top[:, 1] + offset

        lines = np.array(list(zip(common_top, common_bottom)))
        
        return top, bottom, lines
    
    def colouring(self, variable, var_type='categorical', cmap=None, highlight='All'):
        """
        Colour the scatter plot based on a categorical or continuous variable. Not intended to be called directly. Instead, this method is called by the single_layer and multi_layer methods in the Visualisation class.
        variable: The variable to be used for colouring the scatter plot.
        var_type: The type of the variable. Can be 'categorical' or 'continuous'. Default is 'categorical'.
        cmap: The colour map to be used. Accepts a Matplotlib colormap as a string or dict of form {category: color}. Default is 'tab10' for categorical variables; 'Blues' for continuous variables.
        
        Returns:
        colours: The colours to be used for the scatter plot.
        color_dict: The dictionary of colours used for the categories (only if var_type='categorical').
        """
        # print(locals())
        if highlight == 'All':
            if var_type == 'categorical':
                if not isinstance(cmap, dict):
                    cmap = plt.get_cmap(cmap if isinstance(cmap, str) else 'tab10')
                    # variable = variable.dropna()
                    unique_vals = np.unique(variable)
                    color_dict = {val: cmap(i / len(unique_vals)) for i, val in enumerate(unique_vals)}
                    # print(color_dict)
                else:
                    color_dict = {k: matplotlib.colors.to_rgba(v) for k, v in cmap.items()}

                colours = np.array([color_dict[val] for val in variable])
                return colours, color_dict
            elif var_type == 'continuous':
                cmap = plt.get_cmap(cmap if isinstance(cmap, str) else 'Blues')
                norm = plt.Normalize(variable.min(), variable.max())
                colours = cmap(norm(variable))
                return colours, cmap, norm
            else:
                raise ValueError(f"var_type must be either 'categorical' or 'continuous', not {var_type}.")
        else:
            if isinstance(highlight, str):
                highlight = [highlight]
            if var_type == 'categorical':
                if not isinstance(cmap, dict):
                    cmap = plt.get_cmap(cmap if isinstance(cmap, str) else 'tab10')
                    unique_vals = np.unique(variable)
                    color_dict = {val: np.array(cmap(i / len(unique_vals))) for i, val in enumerate(unique_vals)}
                    color_dict['No label'] = matplotlib.colors.to_rgba('black')
                    chosen  = []

                    for i in highlight:
                        chosen.append(i)
                    
                    color_dict = {val: np.array([*color[:-1], 0.1]) if val not in chosen else color for val, color in color_dict.items()}
                    colours = np.array([color_dict[val] for val in variable])
                else:
                    cmap['No label'] = np.array(matplotlib.colors.to_rgba('black'))
                    color_dict = {k: matplotlib.colors.to_rgba(v) for k, v in cmap.items()}
                    chosen  = []

                    for i in highlight:
                        chosen.append(i)
                    
                    color_dict = {val: np.array([*color[:-1], 0.05]) if val not in chosen else color for val, color in color_dict.items()}
                    colours = np.array([color_dict[val] for val in variable])

                return colours, color_dict
            elif var_type == 'continuous':
                cmap = plt.get_cmap(cmap if isinstance(cmap, str) else 'Blues')
                norm = plt.Normalize(variable.min(), variable.max())
                colours = cmap(norm(variable))
                return colours, cmap, norm
            # else:
                raise ValueError(f"var_type must be either 'categorical' or 'continuous', not {var_type}.")

    def single_layer(self, x, y, cvar=None, cvar_type=None, highlight='All', scatter_kwargs=None, cmap=None, interactive=False, drop_outliers=False):
        """
    Create a scatter plot of a single layer. Can be static or interactive. In interactive mode, the user can select the variable to interact with via 'cvar' and 'cvar_type'. If Static, users can highi=light specific groups through 'highlight'.
    
    Args:
    x: The column containing the x-coordinates of the scatter plot.
    y: The column containing the y-coordinates of the scatter plot.
    cvar: The variable to be used for colouring the scatter plot.
    cvar_type: The type of the variable. Can be 'categorical' or 'continuous'.
    highlight: The group to be highlighted. Default is 'All'. This can be changed through interaction with interactive=True.
    cmap: The colour map to be used. Accepts a Matplotlib colormap as a string or dict of form {category: color}. Default is 'tab10' for categorical variables; 'Blues' for continuous variables.
    scatter_kwargs: Additional arguments to be passed to the scatter plot function. Default is None.
        """
        # plt.ioff()
        data = self.data
        if drop_outliers:
            data = Processing().drop_outliers(data, x, y)

        if cvar_type == 'categorical':
            data[cvar] = data[cvar].fillna('No label')
        elif cvar_type == 'continuous':
            if data[cvar].isna().sum() > 0:
                print(f'WARNING: Continuous variable contains missing values. Mapping those to {data[cvar].min() - 10}(the minimum value - 10, you can remove them from the plot using the slider).')
                value = data[cvar].min() -10
                data[cvar] = data[cvar].fillna(value)

        if isinstance(cmap, dict):
            cmap['No label'] = matplotlib.colors.to_rgba('black')

        if cvar is None:
            plt.scatter(data[x], data[y], **(scatter_kwargs or {}))
            plt.axis('off')
            mngr = plt.get_current_fig_manager()
            mngr.resize(1920,1080)
            plt.show()
        else:
            if cvar_type == 'categorical':
                if interactive:
                    fig, ax = plt.subplots()
                    fig.subplots_adjust(right=0.75)
                    possible_values = data[cvar].dropna().unique() 
                    self.POSSIBLE_VALUES = possible_values

                    self.VAR = cvar
                    self.VAR_TYPE = cvar_type
                    current = 'All'

                    ax2 = fig.add_axes([0.75, 0.3, 0.20, 0.5])
                    check = CheckButtons(ax2, possible_values)

                    color_dict = self.colouring(data[cvar],highlight='All', cmap=cmap)[1]

                    for l in check.labels:
                            l.set_color(color_dict[l.get_text()])

                    if not NOTEBOOK:
                                    # Add a button to clear all checkboxes
                        ax_button = fig.add_axes([0.8, 0.1, 0.15, 0.05])  # Position for button
                        button = Button(ax_button, 'Clear All')

                    def update_check(label):
                        if label is None:
                            current = 'All'
                        else:
                            current = [l for l, v in zip(possible_values, check.get_status()) if v]
                            if len(current) == 0 or len(current) == len(possible_values):
                                current = 'All'

                        colors, _ = self.colouring(data[cvar],highlight=current, cmap=cmap)

                        scatter.set_color(colors)
                        # ax.cla()
                        # ax.scatter(data[x], data[y], c=colors, **(scatter_kwargs or {}))
                        # ax.axis('off')
                        plt.draw()

                    def clear_check(event):  
                        check.clear()
                        update_check(None)  # Refresh plot with default settings
                        

                    check.on_clicked(update_check)
                    try:
                        button.on_clicked(clear_check)
                    except:
                        pass
                    scatter = ax.scatter(data[x], data[y], c=self.colouring(data[cvar],highlight='All', cmap=cmap)[0], s=4, **(scatter_kwargs or {}))
                    ax.axis('off')
                    mngr = plt.get_current_fig_manager()
                    mngr.resize(1920,1080)
                    plt.show()
                    # plt.close(fig)
                else:
                    fig, ax = plt.subplots()
                    colours, color_dict = self.colouring(self.data[cvar], cvar_type)
                    scatter = ax.scatter(data[x], data[y], c=colours, **(scatter_kwargs or {}))

                    # Create a legend and position it outside the plot area
                    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) 
                                for label, color in color_dict.items()]
                    ax.legend(handles=handles, title=cvar, bbox_to_anchor=(1.05, 1), loc='upper left')

                    ax.axis('off')
                    # plt.tight_layout()  # Adjust layout to make sure the legend fits
                    mngr = plt.get_current_fig_manager()
                    mngr.resize(1920,1080)
                    plt.show()
                    # print('hi')
                    # plt.pause(5)
                    plt.close(fig)
            elif cvar_type == 'continuous':
                fig, ax = plt.subplots()
                norm = plt.Normalize(self.data[cvar].min(), self.data[cvar].max())
                cmap = plt.get_cmap(cmap)
                
                # Initialize scatter plot with default alpha (transparency)
                scatter = ax.scatter(self.data[x], self.data[y], c=self.data[cvar], cmap=cmap, norm=norm, s=4, **(scatter_kwargs or {}))
                scatter.set_alpha(1)  # Set initial alpha for all points

                # Add color bar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)  # Adjust fraction and pad to make color bar smaller
                cbar.set_label(cvar)

                # Add RangeSlider for range selection
                ax_slider = fig.add_axes([0.1, 0.02, 0.65, 0.03], label='Range Slider')  # Position for slider
                range_slider = RangeSlider(ax_slider, 'Range', self.data[cvar].min(), self.data[cvar].max(), valinit=(self.data[cvar].min(), self.data[cvar].max()))

                def update(val):
                    min_val, max_val = range_slider.val
                    mask = (self.data[cvar] >= min_val) & (self.data[cvar] <= max_val)
                    
                    # This makes the dots outside the range disappear (personal taste)
                    # scatter.set_array(self.data[cvar].where(mask))
                    
                    # Set alpha values: full opacity for in-range points, reduced opacity for out-of-range points
                    scatter.set_alpha(np.where(mask, 1, 0.1))  # 0.3 is an example; adjust as needed
                    
                    # # Update outline for out-of-range points
                    # scatter.set_edgecolor(np.where(mask, 'none', 'lightgray'))  # Out-of-range points have black edges
                    
                    fig.canvas.draw_idle()

                range_slider.on_changed(update)

                ax.axis('off')
                plt.tight_layout()  # Adjust layout to prevent clipping
                mngr = plt.get_current_fig_manager()
                mngr.resize(1920,1080)
                plt.show()
                # plt.pause(5)
                plt.close(fig) 

    def multi_layer(self, top, bottom, cvar=None, cvar_type=None, cmap=None, highlight='All', top_kwargs=None, bottom_kwargs=None, interactive=False, offset=10, drop_outliers=False):
            """
            Create a multi-layer scatter plot. Can be static or interactive. In interactive mode, the user can select the variable to interact with via 'cvar' and 'cvar_type'. If Static, users can highi=light specific groups through 'highlight'.

            Args:
            top (list): The top layer. Input as a list of columns from the object inputted during class initiation in the form [x, y].
            bottom (list): The bottom layer. Input as a list of columns from the object inputted during class initiation in the form [x, y].
            cvar (str): The variable to be used for colouring the layers. Input as the name of the column in the object inputted during class initiation.
            cvar_type (str, 'continuous' or 'categorical'): The type of the variable. Can be 'categorical' or 'continuous'.
            cmap (dict or str): The colour map to be used. Accepts a Matplotlib colormap as a string or dict of form {category: color}. Default is 'tab10' for categorical variables; 'Blues' for continuous variables.
            highlight (str): The group to be highlighted from the cvar. Default is 'All'. This can be changed through interaction with interactive=True.
            top_kwargs (dict): Additional arguments to be passed to plt for the top layer as a dictionary. Default is None.
            bottom_kwargs (dict): Additional arguments to be passed to plt for the bottom layer as a dictionary. Default is None.
            interactive (bool): Whether to create an interactive plot. Default is False.
            offset (int): The distance between the two layers. Default is 10. Dependent on screen size—adjust as needed.
            """
            data = self.data

            if drop_outliers:
                data = Processing().drop_outliers(data, top[0], top[1])
                data = Processing().drop_outliers(data, bottom[0], bottom[1])

            def norm(x):
                return (x - np.min(x)) / (np.max(x) - np.min(x))

            # # Assuming 'top' and 'bottom' are NumPy arrays
            # data[top].iloc[:, 0] = norm(data[top].iloc[:, 0])
            # data[top].iloc[:, 1] = norm(data[top].iloc[:, 1])
            # data[bottom].iloc[:, 0] = norm(data[bottom].iloc[:, 0])
            # data[bottom].iloc[:, 1] = norm(data[bottom].iloc[:, 1]) 



            # Assuming 'top' and 'bottom' are NumPy arrays
            data[top[0]] = norm(data[top[0]])
            data[top[1]] = norm(data[top[1]])
            data[bottom[0]] = norm(data[bottom[0]])
            data[bottom[1]] = norm(data[bottom[1]]) 
                

            if cvar_type == 'categorical':
                data[cvar] = data[cvar].fillna('No label')
            elif cvar_type == 'continuous':
                if data[cvar].isna().sum() > 0:
                        print(f'WARNING: Continuous variable contains missing values. Mapping those to {self.data[cvar].min() - 10}(the minimum value - 10, you can remove them from the plot using the slider).')
                        value = self.data[cvar].min() -10
                        data[cvar] = data[cvar].fillna(value)


            common = data[data[top[0]].notnull() & data[bottom[0]].notnull()]
            print(f'Common: {len(common)}')
            
            top_df = data.dropna(subset=top, how='any')

            bottom_df = data.dropna(subset=bottom, how='any')

            

            top_c, bottom_c, lines = self.align_layers(top_df[top], bottom_df[bottom], common=common, offset=offset)

            if cmap is not None and cvar_type == 'categorical':
                cmap['No label'] = matplotlib.colors.to_rgba('black')

            if cvar is None:
                if interactive:
                    print('WARNING: Interactive mode requires coloring variable cvar. Starting in static mode.')
                fig, ax = plt.subplots()
                coordinate_pairs = list(zip(top_c, bottom_c))
                lines = np.array(coordinate_pairs)
                x_start, y_start = lines[:, 0, 0], lines[:, 0, 1]
                x_end, y_end = lines[:, 1, 0], lines[:, 1, 1]
                lc1 = LineCollection(list(zip(zip(x_start, y_start), zip(x_end, y_end))), color='lightgray', linewidth=0.5)
                ax.add_collection(lc1)


                plt.scatter(top_c[:, 0], top_c[:, 1], **(top_kwargs or {}))
                plt.scatter(bottom_c[:, 0], bottom_c[:, 1], **(bottom_kwargs or {}))
                plt.axis('off')
                mngr = plt.get_current_fig_manager()
                mngr.resize(1920,1080)
                plt.show()
                plt.close(fig)
            else:
                if cvar_type == 'categorical':
                    if interactive:
                        fig, ax = plt.subplots()
                        fig.subplots_adjust(right=0.8)
                        possible_values = self.data[cvar].dropna().unique()
                        self.POSSIBLE_VALUES = possible_values
                        self.VAR = cvar
                        self.VAR_TYPE = cvar_type
                        current = 'All'


                        colours_top, color_dict = self.colouring(top_df[cvar], cvar_type, highlight=highlight, cmap=cmap)

                        color_dict['No label'] = np.array(matplotlib.colors.to_rgba('black')) 
                        colours_bottom, _ = self.colouring(bottom_df[cvar], cvar_type, cmap=color_dict)

                        x_start, y_start = lines[:, 0, 0], lines[:, 0, 1]
                        x_end, y_end = lines[:, 1, 0], lines[:, 1, 1]
                        lc1 = LineCollection(list(zip(zip(x_start, y_start), zip(x_end, y_end))), color=[0.8,0.8,0.8,0.4], linewidth=0.5)
                        ax.add_collection(lc1)


                        ax2 = fig.add_axes([0.8, 0.5, 0.15, 0.3])
                        check = CheckButtons(ax2, possible_values)
                        for l in check.labels:
                            l.set_color(color_dict[l.get_text()])

                        if not NOTEBOOK:
                                            # Add a button to clear all checkboxes
                            ax_button = fig.add_axes([0.8, 0.2, 0.15, 0.05])  # Position for button
                            button = Button(ax_button, 'Clear All')

                        def update_check(label):
                            nonlocal current, cmap
                            if label is None:
                                current = 'All'

                            else:
                                current = [l for l, v in zip(possible_values, check.get_status()) if v]
                                if len(current) == 0 or len(current) == len(possible_values):
                                    current = 'All'

                            top_colors, color_dict = self.colouring(variable=top_df[cvar], var_type=cvar_type, highlight=current, cmap=cmap)
                            bottom_colors, _ = self.colouring(variable=bottom_df[cvar], var_type=cvar_type, highlight=current, cmap=color_dict)


                            top_scatter.set_color(top_colors)
                            bottom_scatter.set_color(bottom_colors)



                                # Update line collection
                            line_segments = []
                            for i in range(len(x_start)):
                                if current == 'All':
                                    line_segments.append(((x_start[i], y_start[i]), (x_end[i], y_end[i])))

                                elif common[cvar].iloc[i] in current:
                                    line_segments.append(((x_start[i], y_start[i]), (x_end[i], y_end[i])))

                            lc1.set_segments(line_segments)
                            lc1.set_alpha(0.6 if len(line_segments) > 0 else 0.1)  # Adjust alpha for lines




                            fig.canvas.draw_idle()
                            plt.draw()
                        
                        def clear_check(event):  
                            check.clear()
                            update_check(None)  # Refresh plot with default settings
                            

                        check.on_clicked(update_check)
                        try:
                            button.on_clicked(clear_check)
                        except:
                            pass

                        top_scatter = ax.scatter(top_c[:, 0], top_c[:, 1], c=colours_top,s=4, zorder=2.5, **(top_kwargs or {}))
                        bottom_scatter = ax.scatter(bottom_c[:, 0], bottom_c[:, 1], c=colours_bottom,s=4, zorder=2.5, **(bottom_kwargs or {}))
                        ax.axis('off')

                        plt.show()
                        # plt.close(fig)

                    else:
                        colours_top, color_dict = self.colouring(top_df[cvar], cvar_type, highlight=highlight, cmap=cmap)
                        colours_bottom, _ = self.colouring(bottom_df[cvar], cvar_type, cmap=color_dict)
                        fig, ax = plt.subplots()


                        x_start, y_start = lines[:, 0, 0], lines[:, 0, 1]
                        x_end, y_end = lines[:, 1, 0], lines[:, 1, 1]
                        lc1 = LineCollection(list(zip(zip(x_start, y_start), zip(x_end, y_end))), color='lightgray', linewidth=0.5)

                        if highlight != 'All':
                            line_segments = []
                            for i in range(len(x_start)):
                                if highlight == 'All':
                                    line_segments.append(((x_start[i], y_start[i]), (x_end[i], y_end[i])))

                                elif common[cvar].iloc[i] in highlight:
                                    line_segments.append(((x_start[i], y_start[i]), (x_end[i], y_end[i])))

                            lc1.set_segments(line_segments)
                            lc1.set_alpha(0.6 if len(line_segments) > 0 else 0.1)



                        ax.add_collection(lc1)

                        plt.scatter(top_c[:, 0], top_c[:, 1], c=colours_top, s=4,zorder=2.5,  **(top_kwargs or {}))
                        plt.scatter(bottom_c[:, 0], bottom_c[:, 1], c=colours_bottom, s=4, zorder=2.5, **(bottom_kwargs or {}))
                        # Create a legend
                        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=np.array([*color[:-1], 1]), markersize=10, label=label) 
                                    for label, color in color_dict.items()]
                        plt.legend(handles=handles, title=cvar, loc='right')
                        plt.axis('off')
                        # mngr = plt.get_current_fig_manager()
                        # mngr.resize(1920,1080)
                        plt.show()
                        # plt.close(fig)

                elif cvar_type == 'continuous':

                    norm = plt.Normalize(data[cvar].min(), data[cvar].max())
                    cmap = plt.get_cmap(cmap if isinstance(cmap, str) else 'Blues')

                    colours_top = cmap(norm(top_df[cvar]))
                    colours_bottom = cmap(norm(bottom_df[cvar]))
                    
                    fig, ax = plt.subplots()

                    # coordinate_pairs = list(zip(top_c, bottom_c))
                    # lines = np.array(coordinate_pairs)
                    x_start, y_start = lines[:, 0, 0], lines[:, 0, 1]
                    x_end, y_end = lines[:, 1, 0], lines[:, 1, 1]
                    lc1 = LineCollection(list(zip(zip(x_start, y_start), zip(x_end, y_end))), color='lightgray', linewidth=0.5)
                    ax.add_collection(lc1)

                    t_scatter = ax.scatter(top_c[:, 0], top_c[:, 1], c=colours_top, s=4, zorder=2.5, **(top_kwargs or {}))
                    b_scatter = ax.scatter(bottom_c[:, 0], bottom_c[:, 1], c=colours_bottom, s=4, zorder=2.5, **(bottom_kwargs or {}))
                    # Add a color bar
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, fraction=0.01, pad=0.01)
                    cbar.set_label(cvar)

                    # Add RangeSlider for range selection
                    ax_slider = fig.add_axes([0.1, 0.02, 0.65, 0.03], label='Range Slider')  # Position for slider
                    range_slider = RangeSlider(ax_slider, f'{cvar}', self.data[cvar].min(), self.data[cvar].max(), valinit=(self.data[cvar].min(), self.data[cvar].max()))

                    def update(val):
                        min_val, max_val = range_slider.val
                        mask_top = (top_df[cvar] >= min_val) & (top_df[cvar] <= max_val)
                        mask_bottom = (bottom_df[cvar] >= min_val) & (bottom_df[cvar] <= max_val)
                        common_mask = (common[cvar] >= min_val) & (common[cvar] <= max_val)
                        
                        # This makes the dots outside the range disappear (personal taste)
                        # scatter.set_array(self.data[cvar].where(mask))
                        
                        # Set alpha values: full opacity for in-range points, reduced opacity for out-of-range points
                        t_scatter.set_alpha(np.where(mask_top, 1, 0.1))  # 0.3 is an example; adjust as needed
                        b_scatter.set_alpha(np.where(mask_bottom, 1, 0.1))  # 0.3 is an example; adjust as needed
                        # # Update outline for out-of-range points
                        # scatter.set_edgecolor(np.where(mask, 'none', 'lightgray'))  # Out-of-range points have black edges
                        
                        # Update LineCollection with transparency
                        line_segments = []
                        for i in range(len(x_start)):
                            if common_mask.iloc[i]:
                                line_segments.append(((x_start[i], y_start[i]), (x_end[i], y_end[i])))
                        
                        lc1.set_segments(line_segments)
                        lc1.set_alpha(1) 


                        fig.canvas.draw_idle()

                    range_slider.on_changed(update)

                    ax.axis('off')
                    # plt.tight_layout()  # Adjust layout to prevent clipping
                    plt.show()
                    # plt.close(fig)
                
    def silhouette(self, group_column, d=None, silhouette_column=None, subset=None, cmap=None, plot_kwargs=None):
        """
        Plot the silhouette scores for each group in the data. Can be called directly using the data imputed when initiating the Visualisation class (using the 'data' parameter and 'subset' if desired) or by the Calculation class with plot=True.

        group_column: The column containing the group labels.
        d: The data to be used if calling this method directly. Default is None.
        silhouette_column: The column containing the silhouette scores. Default is 'silhouette'.
        subset: The subset of groups to be plotted. Default is None. If this method is called through Calculation, the subset is inherited from Calculation.silhouette(..., subset=subset, plot=True).
        """

        if d is None:
            data = self.data
        else:
            data = d
        if silhouette_column is None:
            raise ValueError('silhouette_column must be specified. If you have not calculated silhouette scores, use Calculation(data).silhouette(..., plot=True) instead.')
        # plt.ion()

        if subset is not None:
            data = data[data[group_column].isin(subset)]

        groups = data[group_column].unique()

        if isinstance(cmap, dict):
            colors = {k: matplotlib.colors.to_rgba(v) for k, v in cmap.items()}
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
            colors = {i: cmap(i / len(data[group_column].unique())) for i in data[group_column].unique()}
        else:
            colors = [None for _ in range(len(groups))]

        


        fig,ax = plt.subplots()
        for i, g in enumerate(groups):
            subset_group = data[data[group_column] == g]
            subset_group[silhouette_column].plot(kind='kde', ax=ax, label=g, color=colors[g], **(plot_kwargs or {}))
        
        plt.xlim(-1, 1)
        plt.title('Silhouette Scores')
        plt.legend()
        plt.show()
        # plt.pause(0.001)
        # plt.close(fig)
        
    def overlap(self, densities, x_grid, cmap=None, plot_kwargs=None):
        """
        This method plots 1d densities. Not supposed to be called directly. Instead, this method is called by the overlap method in the Calculation class, using plot=True.
        """
        if isinstance(cmap, dict):
            cmap = {k: matplotlib.colors.to_rgba(v) for k, v in cmap.items()}
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
            cmap = {i: cmap(i / len(densities)) for i in densities.keys()}
        else:
            cmap = plt.get_cmap('tab10')
            cmap = {i: cmap(i / len(densities)) for i in densities.keys()}


        try:
            # plt.ion()
            fig, ax = plt.subplots()
            for cluster in densities:
                ax.plot(x_grid, densities[cluster], label=cluster, color=cmap[cluster], **(plot_kwargs or {}))
            ax.set_title('1d Landscapes')
            ax.legend()
            plt.axis('off')
            plt.show()
            # plt.close(fig)
        except:
            raise ValueError('Did you call this method directly? Try instead calling Calculations(data).overlap(..., plot=True). The function needs the densities to provide the plot.')
        
