import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import combinations


def is_df_equal(lst_dfs, is_raise: bool = True):
    if len(lst_dfs) == 1:
        pass
    else:
        lst0 = [
            i[0].index.equals(i[1].index)
            for i in [
                *combinations(
                    [d.sort_index(axis=0).sort_index(axis=1) for d in lst_dfs], 2
                )
            ]
        ]
        lst1 = [
            i[0].columns.equals(i[1].columns)
            for i in [
                *combinations(
                    [d.sort_index(axis=1).sort_index(axis=1) for d in lst_dfs], 2
                )
            ]
        ]
        if not ((sum(lst0) == len(lst0)) and (sum(lst1) == len(lst1))):
            if is_raise:
                raise KeyError("pd.DataFrames are not equivalent")
            else:
                warnings.warn("pd.DataFrames are not equivalent")

        pass


def get_fig_axes(
    n: int, n_cols: int = 3, length_col: float = 2.0, length_row: float = 3.0
):
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * length_col, n_rows * length_row)
    )
    return fig, axes


def _plot_func_basis(
    dfs: list | dict | pd.DataFrame,
    is_raise: bool = False,
) -> (plt.Figure, plt.Axes, int, dict, bool):
    is_legend = False

    if isinstance(dfs, pd.DataFrame):
        dict_dfs = {0: dfs}

    elif isinstance(dfs, list):
        dict_dfs = dict(zip([*range(len(dfs))], dfs))

    elif isinstance(dfs, dict):
        dict_dfs = dfs.copy()
        is_legend = True

    else:
        raise KeyError(f"{type(dfs)} is not a list or dict")

    is_df_equal(list(dict_dfs.values()), is_raise=is_raise)

    lst = [len(list(d.columns)) for d in dict_dfs.values()]
    index, _ = max(enumerate(lst), key=itemgetter(1))
    lst_df_cols = list(list(dict_dfs.values())[index].columns)

    return dict_dfs, is_legend, lst_df_cols


def get_2d_df_figure(
    dfs: list | dict | pd.DataFrame,
    n_cols: int = 3,
    length_row: float = 2,
    length_col: float = 5,
    flt_rotation: float = 90.0,
    dict_plotting_args: dict = None,
    is_equalise_axis: bool = False,
    flt_smooth_color: float = None,
    fig_title: str = None,
):
    global df_fillbetween_l, df_fillbetween_u
    is_fillbetween = False

    dict_dfs, is_legend, lst_df_cols = _plot_func_basis(dfs, is_raise=True)

    n = list(dict_dfs.values())[0].shape[1]
    fig, axes = get_fig_axes(
        n=n, n_cols=n_cols, length_row=length_row, length_col=length_col
    )

    lst_columns = list(dict_dfs.values())[0].columns

    if dict_plotting_args is not None:
        if "fillbetween" in dict_plotting_args.keys():
            is_fillbetween = True
            str_fillbetween_l, str_fillbetween_u = dict_plotting_args["fillbetween"]

            assert (
                str_fillbetween_l in dict_dfs.keys()
            ), f"{str_fillbetween_l} not in dfs: {list(dict_dfs.keys())}"
            assert (
                str_fillbetween_u in dict_dfs.keys()
            ), f"{str_fillbetween_u} not in dfs: {list(dict_dfs.keys())}"

            df_fillbetween_l, df_fillbetween_u = [
                v
                for k, v in dict_dfs.items()
                if k in [str_fillbetween_l, str_fillbetween_u]
            ].copy()

            del dict_dfs[str_fillbetween_l]
            del dict_dfs[str_fillbetween_u]

    if flt_smooth_color is not None:
        arr_colors = plt.cm.jet(np.linspace(0, flt_smooth_color, len(dict_dfs)))

    lst = [lst_df_cols, axes.ravel()]
    # noinspection PyTypeChecker
    for idx_col, (col, ax) in enumerate(list(map(list, zip(*lst)))):
        if idx_col >= n:
            break

        if is_equalise_axis:
            d = pd.DataFrame({k: v.stack() for k, v in dict_dfs.items()}).stack()
            ax.set_ylim(d.min() * 1.01, d.max() * 1.01)

        for idx_df, (legend, df) in enumerate(dict_dfs.items()):
            ax.tick_params(axis="x", labelrotation=flt_rotation)

            if (idx_col > 0) or (not is_legend):
                legend = "__no label__"

            if col in df.columns:
                if flt_smooth_color is not None:
                    ax.plot(df.loc[:, col], label=legend, color=arr_colors[idx_df])
                else:
                    ax.plot(
                        df.loc[:, col],
                        label=legend,
                    )

        if is_fillbetween:
            ax.fill_between(
                df_fillbetween_l.index,
                df_fillbetween_l.iloc[:, idx_col],
                df_fillbetween_u.iloc[:, idx_col],
                alpha=0.25,
            )

        ax.set_title(lst_columns[idx_col])

    if is_legend:
        fig.legend()
    if fig_title is not None:
        fig.suptitle(fig_title)

    fig.tight_layout()
    return fig, axes


def _plot(df, ax, **kwargs):
    return df.plot(ax=ax)


def get_ndf_plot(
    dfs: list | dict | pd.DataFrame,
    plot_func=_plot,
    n_cols: int = 3,
    length_row: float = 2,
    length_col: float = 5,
    flt_rotation: float = 90.0,
    is_equalise_axis: bool = True,
    fig_title: str = None,
    plot_func_kwargs=None,
):
    dict_dfs, is_legend, _ = _plot_func_basis(
        dfs,
    )

    n = len(dict_dfs)
    fig, axes = get_fig_axes(
        n=n, n_cols=n_cols, length_row=length_row, length_col=length_col
    )

    for idx, ax in enumerate(axes.ravel()):
        if idx >= n:
            break

        if is_equalise_axis:
            d = pd.DataFrame({k: v.stack() for k, v in dict_dfs.items()}).stack()

            ax.set_ylim(d.min() * 1.01, d.max() * 1.01)

        ax.tick_params(axis="x", labelrotation=flt_rotation)

        if plot_func_kwargs is not None:
            plot_func(list(dict_dfs.values())[idx], ax=ax, **plot_func_kwargs)
        else:
            plot_func(
                list(dict_dfs.values())[idx],
                ax=ax,
            )

        ax.set_title(list(dict_dfs.keys())[idx])

    if fig_title is not None:
        fig.suptitle(fig_title)

    fig.tight_layout()
    return fig, axes
