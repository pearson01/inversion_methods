"""
Regression test for a real bug: `bristau.py`'s `InversionParameters` (the
user-facing config) and `inversion_bristau.py`'s `InversionInput` (the
internal config the Gibbs sampler actually reads) are two separate
dataclasses with overlapping fields, and `bristau_function` always passes
every shared hyperparameter field through explicitly via
`dataclasses.replace(...)`.

That means `InversionInput`'s own default for a field is *never reachable*
through the real `bristau_function` entrypoint -- whatever
`InversionParameters` declares always wins, silently, even if someone only
updates one side's default. This is exactly what happened with
`tau_resid_max`: it was changed to `200` on `InversionInput` but left at
`None` on `InversionParameters`, so any run relying on the default (rather
than setting `tau_resid_max` in its .ini file) got `None` regardless, and
crashed inside `sample_tau`'s `np.log(tau_max)`.

This only checks fields where *both* dataclasses declare a default -- e.g.
xprior/rprior/nbasis/... are intentionally required (no default) on
InversionParameters (forcing bristau_function's caller to always supply
them) while only defaulted on InversionInput for two-stage-construction
convenience inside bristau_monthly_dictionaries(); that asymmetry is safe,
since the required side always wins before replace() ever runs. The
dangerous pattern -- caught here -- is when both sides *do* have a default
and they've drifted apart, so whichever one bristau_function's
dataclasses.replace(...) call passes through wins silently, regardless of
which value looks like "the default" from InversionInput's own definition.
"""

import dataclasses

from inversion_methods.bristau.bristau import InversionParameters
from inversion_methods.bristau.inversion_bristau import InversionInput


def test_shared_fields_have_matching_defaults():
    params_fields = {f.name: f for f in dataclasses.fields(InversionParameters)}
    input_fields = {f.name: f for f in dataclasses.fields(InversionInput)}

    shared_names = set(params_fields) & set(input_fields)
    assert shared_names, "expected at least some fields in common between the two config dataclasses"

    mismatches = []
    for name in sorted(shared_names):
        p_field = params_fields[name]
        i_field = input_fields[name]

        p_has_default = p_field.default is not dataclasses.MISSING
        i_has_default = i_field.default is not dataclasses.MISSING

        # Only flag it when *both* sides declare a default and they differ --
        # one side being required (no default) is an intentional, safe
        # pattern in this codebase (see module docstring), not a bug.
        if p_has_default and i_has_default and p_field.default != i_field.default:
            mismatches.append(f"{name}: InversionParameters default={p_field.default!r} != InversionInput default={i_field.default!r}")

    assert not mismatches, (
        "InversionParameters (bristau.py) and InversionInput (inversion_bristau.py) "
        "have mismatched defaults for shared fields -- since bristau_function always "
        "passes these through dataclasses.replace(...) explicitly, InversionInput's "
        "default is unreachable and InversionParameters' silently wins:\n"
        + "\n".join(mismatches)
    )
