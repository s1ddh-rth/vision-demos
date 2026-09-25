# vision-demos

Real-world computer vision demos.

| Project | What it does | Key model |
|---|---|---|
| **[dance_sync](dance_sync/)** | Compares dancers' sync performing the same choreography and computes a similarity metric. | [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[chin_ups](chin_ups/)** | Counts chin-up reps from a clip and times the ascent and descent of each one. | [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[rock_climbing](rock_climbing/)** | Segments bouldering holds, returns which holds the climber used and in what order, and compares attempts at the same route. Judges the send IFSC-style, scores landings and silent feet, checks dyno reach, and runs from a local dashboard. | [`sam3.1`](https://docs.vlm.run/gateway/models/facebook-sam3.1) + [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |
| **[running](running/)** | Measures a runner's cadence, times every foot strike, and averages the knee shape at contact. | [`vitpose-plus-large`](https://docs.vlm.run/gateway/models/usyd-community-vitpose-plus-large) |


[![Dancers with pose overlays on the left and a sync score panel on the right](dance_sync/readme_images/dance_demo_thumbnail.jpg)](dance_sync/)

[![A chin-up at the top of the rep with a pose overlay on the left and a rep-timing panel on the right](chin_ups/readme_images/chin_ups_demo_thumbnail.jpg)](chin_ups/)

[![A completed boulder problem: the left panel holds a card comparing this attempt's sequence of holds against three others, the right panel the segmented route with the holds used lit in order](rock_climbing/readme_images/rock_climbing_demo_thumbnail.jpg)](rock_climbing/)

[![A runner on a treadmill with a pose overlay on the left and a live cadence panel on the right](running/readme_images/running_demo_thumbnail.jpg)](running/)

Each project has its own README with setup and instructions.

## License

[Apache-2.0](LICENSE).
