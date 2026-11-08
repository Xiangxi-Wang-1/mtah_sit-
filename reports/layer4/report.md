# Layer 4 evaluation

Evaluate layer 4 without any ground truth.

    python -m possession.layer4_possession_model.evaluate            # reads cache/layer4

  held-out certain moments  out-of-fold accuracy / AUC on the certain-possession frames, per
                            frame and per control spell, and calibration (Brier score)
  Smart Tagging proxy       AUC on the frames a Smart Tagging phase implies a side, split by
                            whether the ball is seen, and on "stale" frames (ball missing, or no
                            control for > 10 s) - where layers 0-3 are weakest
  complement                where the layers 0-3 read-out disagrees with the proxy, how often
                            the model agrees with it
  timing                    lag at the 24 Smart Tagging turnovers

Writes reports/layer4/report.md.

```
{
 "labelled_frames": 27207,
 "labelled_spells": 600,
 "readout_proxy_agreement": 0.8619411318057975,
 "readout_proxy_agreement_stale": 0.6891176470588235,
 "players": {
  "oof_auc_certain": 0.963570682402706,
  "oof_acc_certain": 0.9270408350792075,
  "oof_spell_acc": 0.9266666666666666,
  "brier": 0.05600196638844339,
  "proxy_auc": 0.9443076940689332,
  "proxy_auc_ball_seen": 0.944375433486002,
  "proxy_auc_ball_missing": 0.882253738081959,
  "proxy_auc_stale": 0.9519358523273496,
  "proxy_acc": 0.9139979454196258,
  "proxy_acc_stale": 0.8926470588235295,
  "right_where_readout_wrong": 0.6714655451310255,
  "turnover_lag_median_s": 2.54,
  "turnover_abs_lag_median_s": 4.82
 },
 "full": {
  "oof_auc_certain": 0.9998613566419895,
  "oof_acc_certain": 0.9950012864336384,
  "oof_spell_acc": 0.9966666666666667,
  "brier": 0.004572505504940026,
  "proxy_auc": 0.8583902837847632,
  "proxy_auc_ball_seen": 0.8623889051738419,
  "proxy_auc_ball_missing": 0.7911919368974584,
  "proxy_auc_stale": 0.6884437010541486,
  "proxy_acc": 0.7633436062352048,
  "proxy_acc_stale": 0.5676470588235294,
  "right_where_readout_wrong": 0.3925913943707538,
  "turnover_lag_median_s": 0.78,
  "turnover_abs_lag_median_s": 4.0200000000000005
 },
 "variant_comparison": {
  "moment model, leaves 15": [
   0.943,
   0.906,
   0.94,
   0.886
  ],
  "labels without passes": [
   0.934,
   0.893,
   0.942,
   0.835
  ],
  "moment model, leaves 31": [
   0.945,
   0.91,
   0.943,
   0.888
  ],
  "context model (30 s, +/-5 s), leaves 31": [
   0.966,
   0.938,
   0.933,
   0.856
  ],
  "blend 50/50 (chosen)": [
   0.964,
   0.927,
   0.944,
   0.893
  ]
 }
}
```
