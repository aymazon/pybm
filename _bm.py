# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals
__all__ = ["absolute_import", "division", "print_function", "unicode_literals"]

import platform
import multiprocessing
if platform.system() == "Linux":
    try:
        multiprocessing.set_start_method("forkserver")
    except (RuntimeError, ValueError):
        pass

import typing as t
import pyrsistent.typing as tp
JsonData = t.Dict[str, t.Any]
__all__.extend(["t", "tp", "JsonData"])

from typing import (  # noqa
    Callable, Optional, NewType, Type, Any, NoReturn, Union, Tuple, Iterable,
    List, Dict, Set, FrozenSet, NamedTuple, cast, overload, no_type_check,
    no_type_check_decorator)
__all__.extend([
    "Callable", "Optional", "NewType", "Type", "Any", "NoReturn", "Union",
    "Tuple", "Iterable", "List", "Dict", "Set", "FrozenSet", "NamedTuple",
    "cast", "overload", "no_type_check", "no_type_check_decorator"
])

from abc import ABCMeta, abstractmethod
__all__.extend(["ABCMeta", "abstractmethod"])

import string, os, sys, re, datetime, calendar, collections, heapq, bisect, array, types, copy, enum, decimal, random, glob, shutil, pickle, sqlite3, zlib, gzip, bz2, zipfile, tarfile, csv, configparser, hashlib, io, time, argparse, logging, logging.config, platform, ctypes, threading, multiprocessing, subprocess, queue, socket, asyncio, signal, mmap, json, base64, binhex, binascii, html, xml, webbrowser, urllib, http, ftplib, poplib, imaplib, nntplib, smtplib, telnetlib, uuid, socketserver, xmlrpc, ipaddress, gettext, locale, cmd, doctest, unittest, warnings, abc, gc, inspect, traceback, importlib, math
import six, gevent, pykka, Pyro4, celery, redis, mock, cffi, cython, cython as cy, ipdb, psutil, requests, redis_lock

__all__.extend([
    "string", "os", "sys", "re", "datetime", "calendar", "collections",
    "heapq", "bisect", "array", "types", "copy", "enum", "decimal", "random",
    "glob", "shutil", "pickle", "sqlite3", "zlib", "gzip", "bz2", "zipfile",
    "tarfile", "csv", "configparser", "hashlib", "io", "time", "argparse",
    "logging", "platform", "ctypes", "threading", "multiprocessing",
    "subprocess", "queue", "socket", "asyncio", "signal", "mmap", "json",
    "base64", "binhex", "binascii", "html", "xml", "webbrowser", "urllib",
    "http", "ftplib", "poplib", "imaplib", "nntplib", "smtplib", "telnetlib",
    "uuid", "socketserver", "xmlrpc", "ipaddress", "gettext", "locale", "cmd",
    "doctest", "unittest", "warnings", "abc", "gc", "inspect", "traceback",
    "importlib", "math", "six", "gevent", "pykka", "Pyro4", "celery", "redis",
    "redis_lock", "mock", "cffi", "cython", "cy", "ipdb", "psutil", "requests"
])

try:
    import syslog, fcntl
    __all__.extend(["syslog", "fcntl"])
except ImportError:
    pass

from functools import update_wrapper, wraps, WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES, total_ordering, cmp_to_key, lru_cache, reduce, partial, partialmethod, singledispatch
__all__.extend([
    "update_wrapper", "wraps", "WRAPPER_ASSIGNMENTS", "WRAPPER_UPDATES",
    "total_ordering", "cmp_to_key", "lru_cache", "reduce", "partial",
    "partialmethod", "singledispatch"
])

from operator import abs, add, and_, attrgetter, concat, contains, countOf, delitem, eq, floordiv, ge, getitem, gt, iadd, iand, iconcat, ifloordiv, ilshift, imatmul, imod, imul, index, indexOf, inv, invert, ior, ipow, irshift, is_, is_not, isub, itemgetter, itruediv, ixor, le, length_hint, lshift, lt, matmul, methodcaller, mod, mul, ne, neg, not_, or_, pos, pow, rshift, setitem, sub, truediv, truth, xor
__all__.extend([
    "abs", "add", "and_", "attrgetter", "concat", "contains", "countOf",
    "delitem", "eq", "floordiv", "ge", "getitem", "gt", "iadd", "iand",
    "iconcat", "ifloordiv", "ilshift", "imatmul", "imod", "imul", "index",
    "indexOf", "inv", "invert", "ior", "ipow", "irshift", "is_", "is_not",
    "isub", "itemgetter", "itruediv", "ixor", "le", "length_hint", "lshift",
    "lt", "matmul", "methodcaller", "mod", "mul", "ne", "neg", "not_", "or_",
    "pos", "pow", "rshift", "setitem", "sub", "truediv", "truth", "xor"
])

from itertools import count, cycle, repeat, accumulate, chain, compress, dropwhile, filterfalse, groupby, islice, starmap, takewhile, tee, zip_longest, product, permutations, combinations, combinations_with_replacement
__all__.extend([
    "count", "cycle", "repeat", "accumulate", "chain", "compress", "dropwhile",
    "filterfalse", "groupby", "islice", "starmap", "takewhile", "tee",
    "zip_longest", "product", "permutations", "combinations",
    "combinations_with_replacement"
])

from collections import deque, defaultdict, namedtuple, UserDict, UserList, UserString, Counter, OrderedDict, ChainMap
__all__.extend([
    "deque", "defaultdict", "namedtuple", "UserDict", "UserList", "UserString",
    "Counter", "OrderedDict", "ChainMap"
])

from pyrsistent import pmap, m as pm, PMap, pvector, v as pv, PVector, pset, s as ps, PSet, pbag, b as pb, PBag, plist, l as pl, PList, pdeque, dq as pdq, PDeque, CheckedPMap, CheckedPVector, CheckedPSet, InvariantException, PTypeError, CheckedKeyTypeError, CheckedValueTypeError, CheckedType, optional, PRecord, field as pfield, pset_field, pmap_field, pvector_field, PClass, PClassMeta, immutable, freeze, thaw, mutant, get_in, inc, discard, rex, ny  # type: ignore
__all__.extend([
    "pmap", "pm", "PMap", "pvector", "pv", "PVector", "pset", "ps", "PSet",
    "pbag", "pb", "PBag", "plist", "pl", "PList", "pdeque", "pdq", "PDeque",
    "CheckedPMap", "CheckedPVector", "CheckedPSet", "InvariantException",
    "PTypeError", "CheckedKeyTypeError", "CheckedValueTypeError",
    "CheckedType", "optional", "PRecord", "pfield", "pset_field", "pmap_field",
    "pvector_field", "PClass", "PClassMeta", "immutable", "freeze", "thaw",
    "mutant", "get_in", "inc", "discard", "rex", "ny"
])

from fn.monad import Option, Full, Empty, optionable
from fn.op import apply, call, zipwith, foldl, foldr, unfold
from fn.stream import Stream
from fn.recur import tco
from fn.iters import padnone, ncycles, repeatfunc, consume
from fn.iters import partition as splitin, splitat, splitby
from fn.iters import powerset, pairwise, iter_except, flatten
__all__.extend([
    "Option", "Full", "Empty", "optionable", "apply", "call", "zipwith",
    "foldl", "foldr", "unfold", "Stream", "tco", "padnone", "ncycles",
    "repeatfunc", "consume", "splitin", "splitat", "splitby", "powerset",
    "pairwise", "iter_except", "flatten"
])

from toolz import remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable, isdistinct, take, drop, take_nth, first, second, nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate, sliding_window, partition, partition_all, count as count_seq, pluck, join, tail, diff, topk, peek, peekn, random_sample, identity, apply, thread_first, thread_last, memoize, compose, compose_left, pipe, complement, juxt, do, curry, excepts, merge, merge_with, valmap, keymap, itemmap, valfilter, keyfilter, itemfilter, assoc, dissoc, assoc_in, update_in, get_in, countby, partitionby, comp, curried as c  # type: ignore
from toolz.sandbox import EqualityHashKey, unzip
from toolz.curried import operator as co
import platform
if platform.python_implementation() == "CPython":
    try:
        from cytoolz import remove, accumulate, groupby, merge_sorted, interleave, unique, isiterable, isdistinct, take, drop, take_nth, first, second, nth, last, get, concat, concatv, mapcat, cons, interpose, frequencies, reduceby, iterate, sliding_window, partition, partition_all, count as count_seq, pluck, join, tail, diff, topk, peek, peekn, random_sample, identity, apply, thread_first, thread_last, memoize, compose, compose_left, pipe, complement, juxt, do, curry, excepts, merge, merge_with, valmap, keymap, itemmap, valfilter, keyfilter, itemfilter, assoc, dissoc, assoc_in, update_in, get_in, countby, partitionby, comp, curried as c  # type: ignore
        from cytoolz.curried import operator as co
    except ImportError:
        pass
c.compress = curry(compress)
c.dropwhile = curry(dropwhile)
c.starmap = curry(starmap)
c.takewhile = curry(takewhile)
c.combinations = curry(combinations)
c.combinations_with_replacement = curry(combinations_with_replacement)
c.freeze = curry(freeze)
c.thaw = curry(thaw)
c.discard = curry(discard)
c.ncycles = curry(ncycles)
c.splitin = curry(splitin)
c.splitat = curry(splitat)
c.splitby = curry(splitby)
c.iter_except = curry(iter_except)

__all__.extend([
    "remove", "accumulate", "groupby", "merge_sorted", "interleave", "unique",
    "isiterable", "isdistinct", "take", "drop", "take_nth", "first", "second",
    "nth", "last", "get", "concat", "concatv", "mapcat", "cons", "interpose",
    "frequencies", "reduceby", "iterate", "sliding_window", "partition",
    "partition_all", "count_seq", "pluck", "join", "tail", "diff", "topk",
    "peek", "peekn", "random_sample", "identity", "apply", "thread_first",
    "thread_last", "memoize", "compose", "compose_left", "pipe", "complement",
    "juxt", "do", "curry", "excepts", "merge", "merge_with", "valmap",
    "keymap", "itemmap", "valfilter", "keyfilter", "itemfilter", "assoc",
    "dissoc", "assoc_in", "update_in", "get_in", "countby", "partitionby",
    "comp", "c", "co", "EqualityHashKey", "unzip"
])

PY35 = sys.version_info[0] == 3 and sys.version_info[1] == 5
PY36 = sys.version_info[0] == 3 and sys.version_info[1] == 6
PY37 = sys.version_info[0] == 3 and sys.version_info[1] == 7
PY38 = sys.version_info[0] == 3 and sys.version_info[1] == 8
from toolz.compatibility import map, filter, range, zip, reduce, zip_longest, iteritems, iterkeys, itervalues, filterfalse, PY3, PY34, PYPY  # type: ignore
__all__.extend([
    "map", "filter", "range", "zip", "reduce", "zip_longest", "iteritems",
    "iterkeys", "itervalues", "filterfalse", "PY3", "PY34", "PY35", "PY36",
    "PY37", "PY38", "PYPY"
])

try:
    from dataclasses import dataclass, field
    __all__.extend(["dataclass", "field"])
except ImportError:
    pass

from functoolsex import flip, F, FF, X, op_filter, op_map, op_or_else, op_or_call, op_get_or, op_get_or_call, R, fold, is_none, is_not_none, is_option_full, is_option_empty, uppack_args, combinations_with_replacement, compress, every, first_object, first_option_full, first_pred_object, first_true, getter, laccumulate, lchain, lcombinations, lcombinations_with_replacement, lcompact, lcompress, lconcat, lconcatv, lcons, lcycle, ldiff, ldrop, ldropwhile, lfilter, lfilterfalse, lflatten, lgrouper, linterleave, linterpose, lislice, liter_except, lmap, lmapcat, lmerge_sorted, lncycles, lpairwise, lpartition, lpartition_all, lpermutations, lpluck, ljoin, lpowerset, lproduct, lrandom_sample, lpartitionby, lrange, lreject, lremove, lrepeat, lrepeatfunc, lrest, lroundrobin, lsliding_window, lsplitat, lsplitby, lsplitin, lstarmap, ltail, ltake, ltake_nth, ltakewhile, ltee, ltopk, lunique, lzip, lzip_longest, taccumulate, tchain, tcombinations, tcombinations_with_replacement, tcompact, tcompress, tconcat, tconcatv, tcons, tcycle, tdiff, tdrop, tdropwhile, tfilter, tfilterfalse, tflatten, tgrouper, tinterleave, tinterpose, tislice, titer_except, tmap, tmapcat, tmerge_sorted, tncycles, tpairwise, tpartition, tpartition_all, tpermutations, tpluck, tjoin, tpowerset, tproduct, trandom_sample, tpartitionby, trange, treject, tremove, trepeat, trepeatfunc, trest, troundrobin, tsliding_window, tsplitat, tsplitby, tsplitin, tstarmap, ttail, ttake, ttake_nth, ttakewhile, ttee, ttopk, tunique, tzip, tzip_longest, some, tco_yield  # type: ignore

__all__.extend([
    "flip", "F", "FF", "X", "R", "fold", "op_filter", "op_map", "op_or_else",
    "op_or_call", "op_get_or", "op_get_or_call", "is_none", "is_not_none",
    "is_option_full", "is_option_empty", "uppack_args",
    "combinations_with_replacement", "compress", "every", "first_object",
    "first_option_full", "first_pred_object", "first_true", "getter",
    "laccumulate", "lchain", "lcombinations", "lcombinations_with_replacement",
    "lcompact", "lcompress", "lconcat", "lconcatv", "lcons", "lcycle", "ldiff",
    "ldrop", "ldropwhile", "lfilter", "lfilterfalse", "lflatten", "lgrouper",
    "linterleave", "linterpose", "lislice", "liter_except", "lmap", "lmapcat",
    "lmerge_sorted", "lncycles", "lpairwise", "lpartition", "lpartition_all",
    "lpermutations", "lpluck", "ljoin", "lpowerset", "lproduct",
    "lrandom_sample", "lpartitionby", "lrange", "lreject", "lremove",
    "lrepeat", "lrepeatfunc", "lrest", "lroundrobin", "lsliding_window",
    "lsplitat", "lsplitby", "lsplitin", "lstarmap", "ltail", "ltake",
    "ltake_nth", "ltakewhile", "ltee", "ltopk", "lunique", "lzip",
    "lzip_longest", "taccumulate", "tchain", "tcombinations",
    "tcombinations_with_replacement", "tcompact", "tcompress", "tconcat",
    "tconcatv", "tcons", "tcycle", "tdiff", "tdrop", "tdropwhile", "tfilter",
    "tfilterfalse", "tflatten", "tgrouper", "tinterleave", "tinterpose",
    "tislice", "titer_except", "tmap", "tmapcat", "tmerge_sorted", "tncycles",
    "tpairwise", "tpartition", "tpartition_all", "tpermutations", "tpluck",
    "tjoin", "tpowerset", "tproduct", "trandom_sample", "tpartitionby",
    "trange", "treject", "tremove", "trepeat", "trepeatfunc", "trest",
    "troundrobin", "tsliding_window", "tsplitat", "tsplitby", "tsplitin",
    "tstarmap", "ttail", "ttake", "ttake_nth", "ttakewhile", "ttee", "ttopk",
    "tunique", "tzip", "tzip_longest", "some", "tco_yield"
])

c.uppack_args = curry(uppack_args)
c.every = curry(every)
c.some = curry(some)
c.lremove = curry(lremove)
c.laccumulate = curry(laccumulate)
c.linterleave = curry(linterleave)
c.ltake = curry(ltake)
c.ltail = curry(ltail)
c.ldrop = curry(ldrop)
c.ltake_nth = curry(ltake_nth)
c.lmapcat = curry(lmapcat)
c.lcons = curry(lcons)
c.linterpose = curry(linterpose)
c.lsliding_window = curry(lsliding_window)
c.lpartition = curry(lpartition)
c.lpartition_all = curry(lpartition_all)
c.lpluck = curry(lpluck)
c.ljoin = curry(ljoin)
c.ltopk = curry(ltopk)
c.lrandom_sample = curry(lrandom_sample)
c.lpartitionby = curry(lpartitionby)
c.lrange = curry(lrange)
c.lcycle = curry(lcycle)
c.lrepeat = curry(lrepeat)
c.lcompress = curry(lcompress)
c.ldropwhile = curry(ldropwhile)
c.ltakewhile = curry(ltakewhile)
c.lmap = curry(lmap)
c.lstarmap = curry(lstarmap)
c.lfilter = curry(lfilter)
c.lfilterfalse = curry(lfilterfalse)
c.lcombinations = curry(lcombinations)
c.lcombinations_with_replacement = curry(lcombinations_with_replacement)
c.lreject = curry(lreject)
c.lncycles = curry(lncycles)
c.lgrouper = curry(lgrouper)
c.lsplitin = curry(lsplitin)
c.lsplitat = curry(lsplitat)
c.lsplitby = curry(lsplitby)
c.liter_except = curry(liter_except)
c.tremove = curry(tremove)
c.taccumulate = curry(taccumulate)
c.tinterleave = curry(tinterleave)
c.ttake = curry(ttake)
c.ttail = curry(ttail)
c.tdrop = curry(tdrop)
c.ttake_nth = curry(ttake_nth)
c.tmapcat = curry(tmapcat)
c.tcons = curry(tcons)
c.tinterpose = curry(tinterpose)
c.tsliding_window = curry(tsliding_window)
c.tpartition = curry(tpartition)
c.tpartition_all = curry(tpartition_all)
c.tpluck = curry(tpluck)
c.tjoin = curry(tjoin)
c.ttopk = curry(ttopk)
c.trandom_sample = curry(trandom_sample)
c.tpartitionby = curry(tpartitionby)
c.trange = curry(trange)
c.tcycle = curry(tcycle)
c.trepeat = curry(trepeat)
c.tcompress = curry(tcompress)
c.tdropwhile = curry(tdropwhile)
c.ttakewhile = curry(ttakewhile)
c.tmap = curry(tmap)
c.tstarmap = curry(tstarmap)
c.tfilter = curry(tfilter)
c.tfilterfalse = curry(tfilterfalse)
c.tcombinations = curry(tcombinations)
c.tcombinations_with_replacement = curry(tcombinations_with_replacement)
c.treject = curry(treject)
c.tncycles = curry(tncycles)
c.tgrouper = curry(tgrouper)
c.tsplitin = curry(tsplitin)
c.tsplitat = curry(tsplitat)
c.tsplitby = curry(tsplitby)
c.titer_except = curry(titer_except)

from pprintpp import pprint as pp
__all__.extend(["pp"])

from pysnooper import snoop
__all__.extend(["snoop"])

try:
    import cProfile
    import pstats
    __all__.extend(["cProfile", "pstats"])
except ImportError:
    pass

import pprofile
__all__.extend(["pprofile"])

try:
    import pyximport
    pyximport.install(reload_support=True, language_level=3)
    __all__.extend(["pyximport"])
except ImportError:
    pass

try:
    import numpy as np
    from numpy import nan, nan as NA
    __all__.extend(["np", "nan", "NA"])
except ImportError:
    pass

try:
    import pandas as pd
    pd.options.display.max_rows = 10
    __all__.extend(["pd"])
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    __all__.extend(["plt"])
except ImportError:
    pass

from mock.mock import Mock, MagicMock, PropertyMock
__all__.extend(["Mock", "MagicMock", "PropertyMock"])

import pytest
__all__.extend(["pytest"])

from pytest_mock import MockFixture
__all__.extend(["MockFixture"])

from pykka import ThreadingActor, ActorProxy, ThreadingFuture
__all__.extend(["ThreadingActor", "ActorProxy", "ThreadingFuture"])

PROJECT_NAME = os.environ.get("PROJECT_NAME", "project")
APP_NAME = os.environ.get("APP_NAME", "app")

if platform.system() == "Windows":
    DEFAULT_ROOT_PATH = f"C:"
else:
    DEFAULT_ROOT_PATH = f"/tmp"

PROFILE_ROOT_PATH = os.environ.get("PROFILE_ROOT_PATH",
                                   f"{DEFAULT_ROOT_PATH}/{PROJECT_NAME}/pp")
LOG_FILE_PATH = os.environ.get(
    "LOG_FILE_PATH",
    f"{DEFAULT_ROOT_PATH}/{PROJECT_NAME}/trace/{APP_NAME}.log")

__all__.extend(["PROJECT_NAME", "APP_NAME"])


def cprofile_print_stats(max_call_num: int = 1,
                         step: int = 1,
                         sort_key: int = 2):
    """ cprofile print stats
    Args:
        sort_key: {-1: "stdname", 0: "calls", 1: "time", 2: "cumulative"}
    """
    def wrapper(func):
        r = get_redis_client()
        key = f"{PROJECT_NAME}.{APP_NAME}.cprofile_print_stats"
        with r.Lock(f"{key}-lock", expire=5):
            key_num = r.get(key)
            if key_num is None:
                r.set(key, '0')
            elif int(key_num) >= max_call_num:
                r.set(key, '0')

        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            r.incr(key)
            call_num = int(r.get(key))
            if call_num > max_call_num or (call_num - 1) % step != 0:
                return func(*args, **kwargs)
            print_title = (' ' * 30 + f"-*-cprofile_print_stats-*-|{call_num}")

            pr = cProfile.Profile()
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()

            print('-' * 100)
            print(print_title)
            print('-' * 100)
            print('')
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
            ps.print_stats()
            print(s.getvalue())
            return result

        return inner_wrapper

    return wrapper


def _pprofile_dump(prof: pprofile.Profile,
                   file_path: str,
                   need_rmtree=True) -> None:
    def _pprofile_copy_files() -> None:
        name: str
        for name, lines in prof.iterSource():
            if not os.path.exists(name):
                continue
            src_file = name
            if not name.startswith("/"):
                dst_file = f"{dir_path}/{name}"
            else:
                dst_file = f"{dir_path}{name}"
            dst_path = os.path.dirname(dst_file)
            os.makedirs(dst_path, exist_ok=True)
            if not os.path.exists(dst_file):
                shutil.copyfile(src_file, dst_file)

    dir_path = os.path.dirname(file_path)
    if need_rmtree:
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)
    with io.open(file_path, 'w', errors="replace") as fp:
        prof.callgrind(fp, relative_path=True)
    _pprofile_copy_files()


def pprofile_dump_stats(max_call_num: int = 1, step: int = 1):
    """ pprofile print stats """
    def wrapper(func):
        r = get_redis_client()
        key = f"{PROJECT_NAME}.{APP_NAME}.pprofile_print_stats"
        with r.Lock(f"{key}-lock", expire=5):
            key_num = r.get(key)
            if key_num is None:
                r.set(key, '0')
            elif int(key_num) >= max_call_num:
                r.set(key, '0')

        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            r.incr(key)
            call_num = int(r.get(key))
            if call_num > max_call_num or (call_num - 1) % step != 0:
                return func(*args, **kwargs)
            print_title = (' ' * 30 + f"-*-pprofile_print_stats-*-|{call_num}")

            prof = pprofile.Profile()
            with prof():
                result = func(*args, **kwargs)
            print('-' * 100)
            print(print_title)
            print('-' * 100)
            print('')
            _pprofile_dump(prof,
                           f"{PROFILE_ROOT_PATH}/cachegrind.out.{call_num}")
            return result

        return inner_wrapper

    return wrapper


def pprofile_dump_statistical_stats(max_call_num: int = 1, step: int = 1):
    """ pprofile dump statistical stats """
    def wrapper(func):
        r = get_redis_client()
        key = f"{PROJECT_NAME}.{APP_NAME}.pprofile_print_statistical_stats"
        with r.Lock(f"{key}-lock", expire=5):
            key_num = r.get(key)
            if key_num is None:
                r.set(key, '0')
            elif int(key_num) >= max_call_num:
                r.set(key, '0')

        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            r.incr(key)
            call_num = int(r.get(key))
            if call_num > max_call_num or (call_num - 1) % step != 0:
                return func(*args, **kwargs)
            print_title = (
                ' ' * 30 +
                f"-*-pprofile_print_statistical_stats-*-|{call_num}")

            prof = pprofile.StatisticalProfile()
            with prof(period=0.001, single=True):
                result = func(*args, **kwargs)
            print('-' * 100)
            print(print_title)
            print('-' * 100)
            print('')
            _pprofile_dump(prof,
                           f"{PROFILE_ROOT_PATH}/cachegrind.out.{call_num}")
            return result

        return inner_wrapper

    return wrapper


__all__.extend([
    "cprofile_print_stats", "pprofile_dump_stats",
    "pprofile_dump_statistical_stats"
])


def synchronized(lock):  # type: ignore
    """ Synchronization decorator """
    def wrapper(f):  # type: ignore
        @wraps(f)
        def inner_wrapper(*args, **kw):  # type: ignore
            with lock:
                return f(*args, **kw)

        return inner_wrapper

    return wrapper


__all__.extend(["synchronized"])

_g_singleton_type_lock = threading.RLock()


class Singleton(type):
    """ Singleton mix class """
    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):  # type: ignore
        if cls not in cls._instances:
            cls._locked_call(*args, **kwargs)
        return cls._instances[cls]

    @synchronized(_g_singleton_type_lock)  # type: ignore
    def _locked_call(cls, *args, **kwargs):  # type: ignore
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,
                                        cls).__call__(*args, **kwargs)


__all__.extend(["Singleton"])

_g_singleton_actor_proxy_type_lock = threading.RLock()


class SingletonActorProxy(type):
    """ SingletonActorProxy mix class """
    _instances = {}  # type: ignore
    _instance_creating = False

    def __call__(cls, *args, **kwargs):  # type: ignore
        if cls not in cls._instances:
            if cls._instance_creating:
                return super(SingletonActorProxy,
                             cls).__call__(*args, **kwargs)
            cls._locked_call(*args, **kwargs)
        return cls._instances[cls]["proxy"]

    @synchronized(_g_singleton_actor_proxy_type_lock)  # type: ignore
    def _locked_call(cls, *args, **kwargs):  # type: ignore
        if cls not in cls._instances and not cls._instance_creating:
            cls._instance_creating = True
            instance = cls.start(*args, **kwargs)  # type: ignore
            cls._instance_creating = False
            cls._instances[cls] = {
                "instance": instance,
                "proxy": instance.proxy(),
            }


__all__.extend(["SingletonActorProxy"])


def global_call_only_once(func):
    """ Global call only once function decorator """
    instances: t.Dict[t.Callable, t.Any] = {}
    instances_lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        if func not in instances:
            with instances_lock:
                if func not in instances:
                    instances[func] = func(*args, **kwargs)
        return instances[func]

    return wrapper


__all__.extend(["global_call_only_once"])


def update_logging(log_file_path: str,
                   log_level: str = "DEBUG",
                   expand_str: str = "") -> None:
    assert log_file_path
    LOG_LEVEL = {
        "": 10,
        "DEBUG": 10,
        "INFO": 20,
        "WARN": 30,
        "ERROR": 40,
        "FATAL": 50,
    }[log_level]

    if '/' or '\\' in log_file_path:
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path))

    proc_name = os.environ["PROC_ID"] if os.environ.get(
        "PROC_ID") else APP_NAME
    default_format = ("%(asctime)s %(levelname)-7s %(name)-10s " + proc_name +
                      " %(filename)-20s %(lineno)-4s ")
    default_format += expand_str
    default_format += " - %(message)s"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": default_format,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": LOG_LEVEL,
                "formatter": "default"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "level": LOG_LEVEL,
                "formatter": "default",
                "maxBytes": 100 * 1024 * 1024,
                "backupCount": 20,
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": LOG_LEVEL,
            },
        },
    }

    logging.config.dictConfig(config)
    logging.getLogger("parso").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("pykka").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("redis_lock").setLevel(logging.WARNING)


__all__.extend(["update_logging"])


@global_call_only_once
def make_logger() -> logging.Logger:
    """ Make global logger """
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    log_level: str = os.environ.get("LOGGING_LEVEL", "DEBUG")
    update_logging(LOG_FILE_PATH, log_level)
    return logging.getLogger(APP_NAME)


LOGGER: logging.Logger = make_logger()
__all__.extend(["LOGGER"])


class Promise:
    """
    Base class for the proxy class created in the closure of the lazy function.
    It's used to recognize promises in code.
    """
    pass


def lazy(func, *resultclasses):
    """
    Turn any callable into a lazy evaluated callable. result classes or types
    is required -- at least one is needed so that the automatic forcing of
    the lazy evaluation code is triggered. Results are not memoized; the
    function is evaluated on every access.
    """
    @total_ordering
    class __proxy__(Promise):
        """
        Encapsulate a function call and act as a proxy for methods that are
        called on the result of that function. The function is not evaluated
        until one of the methods on the result is called.
        """
        __prepared = False

        def __init__(self, args, kw):
            self.__args = args
            self.__kw = kw
            if not self.__prepared:
                self.__prepare_class__()
            self.__prepared = True

        def __reduce__(self):
            return (_lazy_proxy_unpickle,
                    (func, self.__args, self.__kw) + resultclasses)

        def __repr__(self):
            return repr(self.__cast())

        @classmethod
        def __prepare_class__(cls):
            for resultclass in resultclasses:
                for type_ in resultclass.mro():
                    for method_name in type_.__dict__:
                        # All __promise__ return the same wrapper method, they
                        # look up the correct implementation when called.
                        if hasattr(cls, method_name):
                            continue
                        meth = cls.__promise__(method_name)
                        setattr(cls, method_name, meth)
            cls._delegate_bytes = bytes in resultclasses
            cls._delegate_text = str in resultclasses
            assert not (cls._delegate_bytes and cls._delegate_text), (
                "Cannot call lazy() with both bytes and text return types.")
            if cls._delegate_text:
                cls.__str__ = cls.__text_cast
            elif cls._delegate_bytes:
                cls.__bytes__ = cls.__bytes_cast

        @classmethod
        def __promise__(cls, method_name):
            # Builds a wrapper around some magic method
            def __wrapper__(self, *args, **kw):
                # Automatically triggers the evaluation of a lazy value and
                # applies the given magic method of the result type.
                res = func(*self.__args, **self.__kw)
                return getattr(res, method_name)(*args, **kw)

            return __wrapper__

        def __text_cast(self):
            return func(*self.__args, **self.__kw)

        def __bytes_cast(self):
            return bytes(func(*self.__args, **self.__kw))

        def __bytes_cast_encoded(self):
            return func(*self.__args, **self.__kw).encode()

        def __cast(self):
            if self._delegate_bytes:
                return self.__bytes_cast()
            elif self._delegate_text:
                return self.__text_cast()
            else:
                return func(*self.__args, **self.__kw)

        def __str__(self):
            # object defines __str__(), so __prepare_class__() won't overload
            # a __str__() method from the proxied class.
            return str(self.__cast())

        def __eq__(self, other):
            if isinstance(other, Promise):
                other = other.__cast()
            return self.__cast() == other

        def __lt__(self, other):
            if isinstance(other, Promise):
                other = other.__cast()
            return self.__cast() < other

        def __hash__(self):
            return hash(self.__cast())

        def __mod__(self, rhs):
            if self._delegate_text:
                return str(self) % rhs
            return self.__cast() % rhs

        def __deepcopy__(self, memo):
            # Instances of this class are effectively immutable. It's just a
            # collection of functions. So we don't need to do anything
            # complicated for copying.
            memo[id(self)] = self
            return self

    @wraps(func)
    def __wrapper__(*args, **kw):
        # Creates the proxy object, instead of the actual value.
        return __proxy__(args, kw)

    return __wrapper__


def _lazy_proxy_unpickle(func, args, kwargs, *resultclasses):
    return lazy(func, *resultclasses)(*args, **kwargs)


def lru_cache_time(seconds, maxsize=None):
    """
    Adds time aware caching to lru_cache
    """
    def wrapper(func):
        # Lazy function that makes sure the lru_cache() invalidate after X secs
        ttl_hash = lazy(lambda: round(time.time() / seconds), int)()

        @lru_cache(maxsize)
        def time_aware(__ttl, *args, **kwargs):
            """
            Main wrapper, note that the first argument ttl is not passed down.
            This is because no function should bother to know this that
            this is here.
            """
            def wrapping(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapping(*args, **kwargs)

        return update_wrapper(partial(time_aware, ttl_hash), func)

    return wrapper


__all__.extend(["lru_cache_time"])


def get_host_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


__all__.extend(["get_host_ip"])

_g_redis_client: t.Optional[redis.Redis] = None


@global_call_only_once
def get_redis_client() -> redis.Redis:
    """ Get the global redis client """
    global _g_redis_client
    _g_redis_client = redis.Redis.from_url(os.environ.get(
        "REDIS_URI", "redis://127.0.0.1:6379/99"),
                                           decode_responses=True)
    setattr(_g_redis_client, "Lock", partial(redis_lock.Lock, _g_redis_client))
    return _g_redis_client


__all__.extend(["get_redis_client"])
