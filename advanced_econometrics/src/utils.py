import multiprocessing
import tqdm


def run_parallel_wrap(
    func, arguments: list, n_process: int = 4, show_progressbar: bool = True, **kwargs
):
    if n_process == 1:
        return [func(arguments[0])]
    else:
        pool = multiprocessing.Pool(processes=n_process)

        if show_progressbar:
            res = list(
                tqdm.tqdm(
                    pool.imap_unordered(func, arguments), total=len(arguments), **kwargs
                )
            )
        else:
            res = list(pool.imap_unordered(func, arguments))

    return res
