"""Starting point of the RAN control dApps hackathon: write your policy here.

Run it with::

    python3 -m examples.spectrum_dapp --link zmq --transport tcp --encoding-method asn1 \\
        --control -ru rfsim --num-prbs 106 --center-freq 3619200000 --e \\
        --policy examples.hackathon_policy:MyPolicy
"""

from spectrum.policy import BaselinePolicy, Observation, Policy


class MyPolicy(Policy):
    """Return the set of PRBs to block; the PRBs you leave out are released.

    ``obs.prb_power_db`` has one level per PRB (dB, relative to the gNB's FFT counts),
    ``obs.power_db`` one per subcarrier, ``obs.valid`` marks what was measured, and
    ``self.history`` holds the previous observations. State lives on ``self``.
    """

    def update(self, obs: Observation) -> set[int]:
        # TODO: replace this with your own policy.
        return set()


class ReferencePolicy(BaselinePolicy):
    """A working but simple policy to compare against: ``--policy examples.hackathon_policy:ReferencePolicy``."""
