package babelfix;

import babel.evolution.likelihood.AnyTipObservationProcess;
import beast.base.core.Description;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.inference.CalculationNode;

/**
 * Babel 0.4.2's AbstractObservationProcess caches the log tree weight (the
 * expected number of observable traits) and invalidates the cache only when
 * the death rate or the tree changes. Under a relaxed clock, a proposal that
 * changes branch rates alone therefore leaves a stale tree weight and the
 * likelihood is wrong (BEAST's -Dbeast.debug=true check fails on the first
 * rate-quantile move). This subclass also invalidates the cache when the
 * branch rate model is dirty, and requires the branch rate model to be given
 * explicitly: if omitted, the stock class silently falls back to a strict
 * clock while the pattern likelihood uses the relaxed one.
 */
@Description("AnyTipObservationProcess that recomputes its cached tree weight when branch rates change "
        + "(fix for relaxed clocks with Babel's SDollo / ALSTreeLikelihood).")
public class AnyTipObservationProcessRC extends AnyTipObservationProcess {

    @Override
    public void initAndValidate() {
        if (branchRateModeInput.get() == null) {
            throw new IllegalArgumentException(
                    "AnyTipObservationProcessRC: branchRateModel must be given and must be the same "
                    + "branch rate model as the enclosing ALSTreeLikelihood");
        }
        super.initAndValidate();
    }

    @Override
    public boolean requiresRecalculation() {
        BranchRateModel brm = branchRateModeInput.get();
        if (brm instanceof CalculationNode && ((CalculationNode) brm).isDirtyCalculation()) {
            weightKnown = false;
        }
        return super.requiresRecalculation();
    }
}
