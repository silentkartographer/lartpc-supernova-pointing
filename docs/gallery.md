# Gallery: two-point DSNT outputs

Each example shows one processed event in one wire-plane view after the muon and its Bragg peak have been removed.

* **left panel:** predicted direction (magenta) against the true initial direction (green), with the angular error;
* **right panels:** the Point A (tail) and Point B (head) probability maps, with **×** marking each DSNT expectation.

The predicted direction is $`\mathbf{r}_B - \mathbf{r}_A`$. Click any image for full resolution. Where available, *original readout* shows the same prediction drawn on the event before the muon was removed. Predicting two points rather than a slope is what preserves the head–tail orientation; see [Slope](coordinate_transforms.md#slope). The training losses for Points A and B are in [`main/headtail/training.py`](../main/headtail/training.py) (ray-geometry and angular terms) and [`main/headtail/losses.py`](../main/headtail/losses.py) (bremsstrahlung terms).

<!-- GALLERY:START -->
<table><tr><td valign="top"><a href="../assets/gallery/event_01771_Z.png"><img src="../assets/gallery/event_01771_Z.png" width="100%" alt="Event 01771 · Z view"></a><br><b>Event 01771 · Z view</b> · <a href="../assets/gallery/event_01771_Z_original.png">original readout</a><br><sub>Clean single-track Michel electron. The tail and head heatmaps overlap but peak at different places, and the predicted direction is 1.2° from truth.</sub></td></tr></table>
<!-- GALLERY:END -->

---

[Return to the project overview](../README.md) · [Coordinate transformations](coordinate_transforms.md)
