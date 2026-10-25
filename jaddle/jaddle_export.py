"""Saving problems and JAX functions to disk (functions via ``jax.export``).

The shared file format behind ``jaddle_convex.save_cp`` / ``save_solver`` and
``jaddle_linear.save_lp`` / ``save_solver``. A file is an ``.npz`` holding a
JSON ``meta`` entry, any serialized functions (``fn_*`` entries, as uint8
arrays) and plain arrays. It loads with ``allow_pickle=False``.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import export

from jaddle.jaddle_basic_types import LPValues, SaddleState

# File format; bump when the layout changes.
EXPORT_FORMAT = 1

# Which loader reads each kind of file, for error messages.
_LOADERS = {
    "cp": "jaddle_convex.load_cp",
    "cp_solver": "jaddle_convex.load_solver",
    "lp_solver": "jaddle_linear.load_solver",
    "lp": "jaddle_linear.load_lp",
}


def register(cls, name):
    """Let exported functions take and return the NamedTuple ``cls``."""
    try:
        export.register_namedtuple_serialization(cls, serialized_name=name)
    except ValueError:  # already registered (module reloaded)
        pass


register(SaddleState, "jaddle.SaddleState")
register(LPValues, "jaddle.LPValues")


def serialized(fn, args, platforms, vjp_order):
    """``fn`` exported at ``args`` (ShapeDtypeStructs), as a uint8 array."""
    exp = export.export(jax.jit(fn), platforms=platforms)(*args)
    try:
        blob = exp.serialize(vjp_order=vjp_order)
    except Exception as exc:
        if vjp_order:
            exc.add_note(
                f"Saving with vjp_order={vjp_order} exports the function's "
                "gradient, which must be pure JAX. make_solver's solve loop has "
                "no gradient, and jaddle_linear.make_solution's runs on the "
                "host (scipy), so save those with vjp_order=0."
            )
        raise
    return np.frombuffer(blob, dtype=np.uint8)


def write(path, kind, meta, arrays):
    """An ``.npz`` at ``path`` exactly (``np.savez`` alone appends ``.npz``)."""
    from jaddle import __version__

    meta = dict(format=EXPORT_FORMAT, kind=kind, jaddle_version=__version__, **meta)
    with open(path, "wb") as f:
        np.savez(f, meta=np.array(json.dumps(meta)), **arrays)


def read(path, kind):
    """``(meta, arrays, exported)`` of a file from ``write``: the ``fn_*``
    entries deserialized (keyed without the prefix), the rest as numpy
    arrays."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {k: data[k] for k in data.files if k != "meta"}
    if meta.get("format") != EXPORT_FORMAT:
        raise ValueError(
            f"{path}: unsupported file format {meta.get('format')!r} "
            f"(this jaddle reads format {EXPORT_FORMAT})"
        )
    if meta.get("kind") != kind:
        loader = _LOADERS.get(meta.get("kind"))
        hint = f"; read it with {loader}" if loader else ""
        raise ValueError(f"{path} holds a {meta.get('kind')!r}, not a {kind!r}{hint}")
    exported = {
        k[len("fn_") :]: export.deserialize(bytearray(v.tobytes()))
        for k, v in arrays.items()
        if k.startswith("fn_")
    }
    for exp in exported.values():
        for aval in exp.in_avals:
            _check_x64(path, aval.dtype)
    return meta, arrays, exported


def _check_x64(path, dtype):
    if jax.dtypes.issubdtype(dtype, jax.dtypes.extended):
        return  # e.g. a typed PRNG key
    if np.dtype(dtype).itemsize == 8 and not jax.config.jax_enable_x64:
        raise ValueError(
            f"{path} was saved with {np.dtype(dtype).name} inputs: enable "
            "jax.config.update('jax_enable_x64', True) before loading it"
        )


def call_in_saved_dtypes(exp, args):
    """``exp.call(*args)`` with each input cast to its saved dtype and each
    floating output cast back to the inputs' float dtype, so a float32 file
    runs under x64 and derivatives flow through the casts both ways."""
    leaves, tree = jax.tree.flatten(args)
    if len(leaves) != len(exp.in_avals):
        raise ValueError(f"expected arguments with structure {exp.in_tree}, got {tree}")
    leaves = [jnp.asarray(leaf) for leaf in leaves]
    floats = [leaf.dtype for leaf in leaves if jnp.issubdtype(leaf.dtype, jnp.inexact)]
    cast = [
        leaf if leaf.dtype == aval.dtype else leaf.astype(aval.dtype)
        for leaf, aval in zip(leaves, exp.in_avals)
    ]
    out = exp.call(*jax.tree.unflatten(tree, cast))
    if not floats:
        return out
    out_dtype = jnp.result_type(*floats)
    return jax.tree.map(
        lambda o: o.astype(out_dtype) if jnp.issubdtype(o.dtype, jnp.inexact) else o,
        out,
    )


def _spec(leaf):
    if not isinstance(leaf, jax.ShapeDtypeStruct):
        leaf = jnp.asarray(leaf)
    return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)


def save_function(fn, path, args, kind, *, batch, platforms, vjp_order):
    """Export ``fn(*args)``, or ``jax.vmap(fn)`` over a leading axis of every
    argument with ``batch``, to ``path``. See ``jaddle_convex.save_solver``."""
    if not args:
        raise ValueError("pass example arguments for fn (e.g. its params)")
    specs = jax.tree.map(_spec, tuple(args))
    if batch:
        # BCOO's batching rules compare the batch size with 1, which a
        # symbolic size cannot answer, so the file takes batches of 2 or
        # more; the loader pads a batch of 1.
        (b,) = export.symbolic_shape("b", constraints=("b >= 2",))
        specs = jax.tree.map(lambda s: jax.ShapeDtypeStruct((b, *s.shape), s.dtype), specs)
        fn = jax.vmap(fn)
    arrays = {"fn_main": serialized(fn, specs, platforms, vjp_order)}
    write(path, kind, dict(batch=bool(batch), num_args=len(args)), arrays)


def load_function(path, kind):
    """The function a ``save_function`` file holds, as ``f(*args)``."""
    meta, _, fns = read(path, kind)
    exp = fns["main"]
    if not meta["batch"]:
        return lambda *args: call_in_saved_dtypes(exp, args)

    def batched(*args):
        leaves = jax.tree.leaves(args)
        if leaves and jnp.shape(leaves[0])[:1] == (1,):
            # A batch of 1, padded to the file's minimum batch of 2.
            doubled = jax.tree.map(lambda a: jnp.concatenate([a, a]), args)
            out = call_in_saved_dtypes(exp, doubled)
            return jax.tree.map(lambda o: o[:1], out)
        return call_in_saved_dtypes(exp, args)

    return batched
