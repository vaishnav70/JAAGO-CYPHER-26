# Jaago — Stay Awake, Stay Safe

**CYPHER '26 (DIGIWEEK '26), J.C. Bose University of Science & Technology, YMCA Faridabad**
**Theme 04 — Intelligent Mistake Prevention & Systemic Safety**

> **It stops the accident before it happens, not after it.**

Jaago is an on-device fatigue and distraction guard. A laptop or phone camera watches for eye closure, yawning and looking away, and fires an escalating alarm before a microsleep continues. Video never leaves the device, and the system works locally in the browser.

**Team — TEAM CHOLE BHATURE**
- Yash — Frontend & Live Demo
- Sarthak — Detection & Logic
- Vaishnav — Product & Pitch

---

## The problem

Fatigue does not announce itself. A driver, night-shift nurse or factory operator can drift into a microsleep in seconds, and by the time anyone notices, it is already too late.

Reported data used here (corridor data only, not a national fatigue share):

- Agra–Lucknow Expressway: of 7,024 crashes from Jan 2021 to Sep 2025, 54.71% were linked to the driver falling asleep, in data presented to the Supreme Court Road Safety Committee in Oct 2025.
- For scale only: 1,77,175 road fatalities were reported in India in 2024 (MoRTH data reported in Parliament). This shows the scale of road safety, not the fatigue share.
- Honest caveat: MoRTH discontinued collecting sleep/fatigue accidents in 2017, so no national fatigue percentage exists. Jaago does not claim to reduce accidents — an 8-hour prototype cannot prove causal reduction.

---

## What Jaago does

### Drowsiness detection
Sustained eye closure for 1.5 seconds (adjustable) triggers a full-width red **DROWSY** alarm. Press **I'm awake** to stop it instantly. A normal blink (100–400 ms) never triggers an alarm.

The live panel shows Eye Openness for both eyes, a live Eyes Closed timer (`0.0s / 1.5s`), and raw MediaPipe values (`eyeBlinkLeft`, `eyeBlinkRight`, `jawOpen`, `gaze`) so the detection is transparent, not a black box.

### Distraction detection
Looking left, right, up or down for a sustained period triggers a **Distracted** warning. The screen shows a live head-direction display — arrow + LEFT / RIGHT / UP / DOWN / CENTRE, with live yaw and pitch values. The logic tracks relative head position from a baseline centre captured at session start, so neutral posture is not confused with drift.

The app uses head pose and gaze together, smooths them over several frames, and holds the distracted state if the face is briefly lost on a far turn. It never flips back to Focused on a single invalid frame.

### Escalating sounds
Sounds are different for distraction and drowsiness, so they can be told apart by ear:

- **Distraction:** a medium, medium-long beep for 3 seconds, then an up-down, long, high-volume beep that loops until the user faces front or presses **I'm back**.
- **Drowsiness:** a medium-loud beep for the first 3 seconds, then a high-volume up-down siren (with vibration where supported) that loops until **I'm awake** is pressed. It never stops by itself and it is impossible to dismiss silently.

Separate test buttons for distraction sound and drowsy sound are in Settings, so each sound can be tested without faking an event.

### Alertness score
A single hero score, 0–100, falls on drowsy, distracted and yawn events and recovers slowly when focused. Every score is shown with its contributors underneath:

- Eyes Open % — Good / Watch / Low
- Head Pose — Good / Watch / Low
- Blinks / min — Good / Watch / Low
- Look-away Count — Good / Watch / Low

The chip refers to that contributor's effect on alertness, not just whether the raw number looks good.

**7-Stage History:** the small bar strip next to the score is the last 7 stages of alertness, not the current score. Green = a Focused stage, amber = a Distracted / Watch stage, red = a Drowsy stage.

### Session timeline
Every event, newest first, with time, status pill, full event text, duration and acknowledge action. Both the start of an episode and its clearing are logged, for example:

- `Distracted — Looking LEFT`
- `Distraction cleared — I'm back`
- `Drowsy — Eyes closed`
- `Drowsy cleared — I'm awake`

Filter the timeline with the segmented control: **All / Focused / Distracted / Drowsy**. Each filter shows its real count, and the list shows only rows matching that status. Cleared rows appear under Focused, and active alarms get an immediate acknowledge action.

### Pre-shift check
A 30-second check before a session: camera, face, lighting and baseline capture, shown as Step x of 4 with seconds remaining. The result is FIT / CAUTION / NOT FIT. This is a prototype indicator, not a medical verdict.

**Set as Centre:** your natural front head position is captured as centre at the start of a session, and can be re-captured any time with the Set as Centre button. This matters because a laptop camera sees the head at different positions depending on posture, seat height and screen placement.

### Safe Rest Assistant (new feature)
Jaago now includes a dedicated **Safe Rest Assistant** that responds to repeated fatigue events in the timeline.

- If there are 2 or more drowsy events or 3 or more distracted events, a rest card appears automatically.
- Users can select a rest duration preset (10 / 15 / 20 minutes).
- A countdown starts, and the user can end rest early if needed.
- After the timer ends, a post-break check prompts the operator to confirm they are ready to continue.
- Once completed, the card collapses into a compact “Rest completed” status.

This adds a practical recovery step to the active vigilance system instead of only warning in the moment.

### Journey Tracker (new feature)
A new **Current Journey** panel tracks trip duration, distance and fatigue events in-session.

- Start time and elapsed drive time
- Distance traveled (GPS when available)
- Fatigue events count
- Rest advisory trigger based on:
  - 2+ hours of continuous driving
  - 200+ km traveled
  - 3+ fatigue events

If geolocation is unavailable, the app falls back to a time-based journey model and shows a transparent notice instead of pretending GPS is working.

### Glasses Mode & Eye Calibration (new feature)
A new settings feature allows a user to compensate for glasses or lens glare.

- **Glasses Mode** reduces false eye-closure readings caused by reflections and frames.
- **Calibrate Eyes (2s)** captures a fresh baseline of eye openness so thresholds stay realistic for the current user.
- This improves reliability in real-world use with different lighting conditions and face setups.

### Demo Mode (Simulated)
A clean, step-by-step simulated episode that works without a camera, in stages: Focused → Distracted → Drowsy → I'm awake. Every Demo Mode screen, timeline row and score carries the word **Simulated**.

Demo controls stay visible without scrolling (fixed control bar while Demo Mode is on, plus Right Arrow = Next stage, Left Arrow = Previous stage, Esc = Back to Live Demo). **Back to Live Demo** turns off the simulated flow and returns the interface to its live state.

### Privacy by design
Camera video is processed in the browser only. No upload, no recording, no backend, no account.

---

## How it works

MediaPipe Tasks Vision Face Landmarker is used for eye-blink, jaw-open and gaze blendshapes, with landmark eye-aspect-ratio as a fallback. Jaago's own logic turns those signals into timed episodes, a score model and a session timeline. Everything is computed locally on-device, with a simple UI designed to explain what is happening instead of hiding it behind a black-box "AI" label.

---

## How to run it

No build step. Do not just double-click the HTML file — the camera and model can be blocked that way.

```bash
# From the project folder:
python3 -m http.server 8000
# Then open in Chrome or Edge:
# http://localhost:8000
```

---

## Extra feature summary

This version adds:
- Safe Rest Assistant after repeated fatigue events
- Journey Tracker for trip duration, distance and rest advisories
- Glasses Mode and Eye Calibration for real-world camera reliability
- Improved session state and lifecycle handling
- More transparent Settings controls for sound, thresholds and camera selection

These additions make Jaago more than a simple alerting prototype — it becomes a fuller workplace safety and vigilance-assist system.
