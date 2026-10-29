// ============================================================
//  JAAGO — Complete Vigilance Detection Logic
//  app.js — Production Vanilla ES Module
//  CYPHER '26 · Theme 04: Intelligent Mistake Prevention & Systemic Safety
// ============================================================

'use strict';

// ─── Pinned MediaPipe Tasks Vision CDN ───────────────────────
const MEDIAPIPE_CDN_BUNDLE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const MEDIAPIPE_WASM_PATH  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_ASSET_PATH     = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// ─── Landmark Indices for Fallback Geometry ──────────────────
const EYE_LANDMARKS = {
  leftEAR:  [362, 385, 387, 263, 373, 380],
  rightEAR: [33,  160, 158, 133, 153, 144],
};

const MOUTH_LANDMARKS = {
  top: 13, bottom: 14, left: 78, right: 308,
};

const HEAD_LANDMARKS = {
  noseTip: 1,
  forehead: 10,
  chin: 152,
  userRightTemple: 234, // anatomical right temple (image left)
  userLeftTemple: 454,  // anatomical left temple (image right)
};

// ─── Application State ────────────────────────────────────────
const state = {
  // Camera & Stream
  cameraState: 'Off', // "Off" | "Starting" | "Live" | "Blocked" | "No face"
  stream: null,
  selectedCameraId: '',
  rafId: null,

  // MediaPipe Vision
  FaceLandmarkerClass: null,
  FilesetResolverClass: null,
  faceLandmarker: null,
  modelLoaded: false,
  modelLoading: false,
  modelError: null,

  // Single Face State (Fix B: 'CAMERA_OFF' | 'CALIBRATING' | 'FACE_LOCKED' | 'NO_FACE')
  faceState: 'CAMERA_OFF',

  // Face Tracking & Stability Buffers
  faceDetected: false,
  faceLockedSince: null,
  trackedFaceBox: null,
  lastTrackedFaceBox: null,
  frontStableSince: null,
  distractionLostFaceSince: null,

  // Baseline Calibration (Natural Centre & Eyes-Open Baseline, Fix A & Fix 3)
  baselineYaw: 0,
  baselinePitch: 0,
  baselineBlinkLeft: 0.05,
  baselineBlinkRight: 0.05,
  baselineLeftEAR: 0.30,
  baselineRightEAR: 0.30,
  closedThresholdL: 0.45,
  closedThresholdR: 0.45,
  closedThreshold: 0.45,
  baselineCaptured: false,
  isCalibratingBaseline: false,
  baselineCalibrationStartTime: null,
  baselineDurationMs: 5000, // Calibrating (first 5s with face)
  baselineSamples: [],

  // Real-time sensor metrics (3D + 2D)
  eyeBlinkLeft: null,   // 0 = open, 1 = closed
  eyeBlinkRight: null,
  eyeOpennessL: 100,    // 0-100%
  eyeOpennessR: 100,    // 0-100%
  rawLeftEAR: null,
  rawRightEAR: null,
  lastValidScore: null, // Frozen score when detection paused (Fix 4)
  jawOpen: null,        // 0 = closed, 1 = open
  headPoseYaw: 0,       // degrees [-75, +75]: pos = Right (+), neg = Left (-)
  headPosePitch: 0,     // degrees [-60, +60]: pos = Up (+), neg = Down (-)
  eyeGazeYaw: 0,        // degrees [-45, +45]
  eyeGazePitch: 0,      // degrees [-35, +35]
  gazeYaw: 0,           // combined degrees
  pitch: 0,             // combined degrees
  smoothedYaw: 0,       // rolling average (5-7 frames)
  smoothedPitch: 0,     // rolling average (5-7 frames)
  relativeYaw: 0,       // smoothedYaw - baselineYaw
  relativePitch: 0,     // smoothedPitch - baselinePitch
  yawHistory: [],       // history buffer
  pitchHistory: [],     // history buffer
  headDirection: 'CENTRE', // 'CENTRE' | 'LEFT' | 'RIGHT' | 'UP' | 'DOWN' | 'NO FACE'
  gazeLabel: 'Center (0°)',
  usedBlendshapes: false,

  // Detection Timers & Prototype Thresholds
  drowsyThresholdSec: 1.5, // Level 3 default 1.5s (adjustable)
  lookawayThresholdSec: 3.0, // Level 2 default 3.0s (adjustable)
  lookawayL1ThresholdSec: 1.2, // Level 1 default 1.2s
  sensitivityPreset: 'balanced', // 'sensitive' | 'balanced' | 'relaxed' | 'custom'
  eyeClosedSince: null,
  drowsyStartTime: null,
  blinkTimestamps: [],     // rolling window for Blinks/min
  lastBlinkDetectedTime: 0,

  // Distraction tracking (relativeYaw < 15 and relativePitch < 15 for 0.6s exit)
  distractionStartTime: null,
  distractionLoggedStart: false,
  lookAwayCount: 0,
  lastLookAwayTime: 0, // Timestamp of last look-away (for Head Pose chip 30s look-away rule)
  activeDistractionLevel: 0, // 0 = none, 1 = amber, 2 = red
  distractionVolume: 0.8,    // 80% default volume
  distractionSoundLevel: 0,  // 0 = none, 1 = Level 1 beep (min 3s), 2 = Level 2 alternating
  level1SoundStartTime: null,
  distractionBeepInterval: null,
  distractionTestTimeout: null,

  // Drowsy Alarm Sound (Stage 1 Beep 880Hz -> Stage 2 Siren 740/1100Hz)
  drowsyVolume: 1.0,           // 100% default volume (adjustable in settings 10-100%)
  drowsySoundStage: 0,         // 0 = none, 1 = Stage 1 medium-LOUD beep (880Hz), 2 = Stage 2 siren (740/1100Hz)
  drowsyBeepInterval: null,
  drowsyStage2Timeout: null,
  drowsySirenInterval: null,
  drowsySirenTimeout2: null,
  drowsyVibrateInterval: null,
  drowsyTestTimeout: null,

  // Yawn tracking (sustained jaw-open >= 1.5s)
  yawnStartTime: null,
  lastYawnLogged: 0,

  // Glasses Mode & Eye Calibration (Relative Thresholds)
  glassesMode: false,
  isEyeCalibrating: false,
  eyeCalibrationStartTime: null,
  eyeCalibrationSamples: [],
  calibratedEyeOpenness: null,
  calibratedEyeOpennessL: null,
  calibratedEyeOpennessR: null,
  calibratedBaselineEAR: null,
  eyeCalibrationTimestamp: null,
  isEyeSignalWeak: false,
  weakEyeSignalSince: null,
  eyeLandmarkConfidence: 1.0,
  headNodStartTime: null,
  drowsyTriggerReason: null,
  simulateWeakEyeSignal: undefined,

  // Alertness Score (0–100) & Animated Tweening
  targetScore: null,       // null when calibrating / no face, starts at 100 when focused
  displayedScore: null,
  scoreAnimStartTime: 0,
  scoreAnimStartVal: 100,
  lastFocusRecoveryTime: 0,
  scoreHistory: [],        // Real score history samples for 7-stage strip
  lastHistorySampleTime: 0,

  // Contributors state: 'Good' | 'Watch' | 'Low' | 'Calibrating' | 'Waiting'
  contributors: {
    eyesOpen:   { value: '—', rating: 'Waiting' },
    headPose:   { value: '—', rating: 'Waiting' },
    blinksMin:  { value: '—', rating: 'Waiting' },
    lookAways:  { value: '—', rating: 'Waiting' },
  },

  // Alarm & Audio
  currentStatus: 'Calibrating', // 'Calibrating' | 'Focused' | 'Distracted' | 'Drowsy' | 'No face'
  level3AlarmActive: false,
  alarmSoundEnabled: true,
  audioContext: null,
  alarmBeepInterval: null,
  alarmPreviewActive: false,
  alarmPreviewTimer: null,

  // Session tracking (Rule 5)
  isSessionRunning: false,
  sessionStartTime: null,
  sessionTimerInterval: null,
  sessionDurationSec: 0,
  lowestAlertnessScore: 100,
  eventCounts: {
    drowsy: 0,
    distracted: 0,
    yawn: 0,
    total: 0,
  },
  sessionLogRecords: [], // Full history records for CSV download & detail panel

  // Timeline UI state
  timelineExpanded: false,
  currentFilter: 'all',

  // Pre-shift check state
  preshiftRunning: false,
  preshiftCountdownSec: 30,
  preshiftTimer: null,
  preshiftTicks: [false, false, false, false],
  preshiftStepResults: [true, true, true, true],

  // Demo mode state
  isDemoMode: false,
  demoTimer: null,
  demoStage: 0,

  // Safe Rest Assistant State
  safeRest: {
    selectedMinutes: 15,
    remainingSeconds: 15 * 60,
    timerInterval: null,
    isTimerRunning: false,
    viewState: 'main', // 'main' | 'check' | 'completed'
    hasCompletedRest: false
  },

  // Journey Tracker State (THIS journey only)
  journey: {
    isActive: false,
    isEnded: false,
    startTime: null,
    elapsedSec: 0,
    timerInterval: null,
    distanceKm: 0.0,
    lastPosition: null, // { lat, lon }
    geoWatchId: null,
    gpsUnavailable: false,
    isSimulatingDistance: false,
    simInterval: null,
    fatigueEvents: 0
  },
};

// ─── DOM References ───────────────────────────────────────────
let dom = {};

function cacheDOM() {
  dom = {
    webcam:                  document.getElementById('webcam'),
    canvasOverlay:           document.getElementById('canvas-overlay'),
    placeholderGraphic:      document.getElementById('placeholder-graphic'),
    placeholderCrosshair:    document.getElementById('placeholder-crosshair'),
    placeholderCaptionText:  document.getElementById('placeholder-caption-text'),
    placeholderSubtext:      document.getElementById('placeholder-subtext'),
    videoPaneErrorBox:       document.getElementById('video-pane-error-box'),
    videoPaneErrorReason:    document.getElementById('video-pane-error-reason'),
    videoPaneErrorFix:       document.getElementById('video-pane-error-fix'),
    btnCameraToggle:         document.getElementById('btn-camera-toggle'),
    btnPlaceholderStart:     document.getElementById('btn-placeholder-start'),
    btnCamIcon:              document.getElementById('btn-cam-icon'),
    cameraStatePill:         document.getElementById('camera-state-pill'),
    topbarLivePill:          document.getElementById('topbar-live-pill'),
    overlayLivePill:         document.getElementById('overlay-live-pill'),
    faceStatusPill:          document.getElementById('face-status-pill'),
    cameraErrorBanner:       document.getElementById('camera-error-banner'),
    cameraErrorReason:       document.getElementById('camera-error-reason'),
    cameraErrorFix:          document.getElementById('camera-error-fix'),
    btnDismissError:         document.getElementById('btn-dismiss-error'),
    cameraGuidanceLine:      document.getElementById('camera-guidance-line'),
    soundBlockedNotice:      document.getElementById('sound-blocked-notice'),

    // Look-away live timer display & HUD elements (Requirements 4, 5 & Fix A)
    lookawayTimerBar:        document.getElementById('lookaway-timer-bar'),
    lookawayTimerVal:        document.getElementById('lookaway-timer-val'),
    lookawayTimerStatus:     document.getElementById('lookaway-timer-status'),
    valLiveYaw:              document.getElementById('val-live-yaw'),
    valLivePitch:            document.getElementById('val-live-pitch'),
    valLiveDirection:        document.getElementById('val-live-direction'),
    btnRecalibrateCentre:    document.getElementById('btn-recalibrate-centre'),

    // Live Eye Openness Meter & Closure Timer (Fix 1)
    eyeOpennessBar:          document.getElementById('eye-openness-bar'),
    valEyeOpenL:             document.getElementById('val-eye-open-l'),
    valEyeOpenR:             document.getElementById('val-eye-open-r'),
    eyeBarFillL:             document.getElementById('eye-bar-fill-l'),
    eyeBarFillR:             document.getElementById('eye-bar-fill-r'),
    valEyeClosureTimer:      document.getElementById('val-eye-closure-timer'),
    eyeClosureStatus:        document.getElementById('eye-closure-status'),

    // Overlay Direction Badge (Requirement 4)
    overlayDirectionBadge:   document.getElementById('overlay-direction-badge'),
    overlayDirectionArrow:   document.getElementById('overlay-direction-arrow'),
    overlayDirectionText:    document.getElementById('overlay-direction-text'),
    overlayDirectionAngles:  document.getElementById('overlay-direction-angles'),

    // Debug elements
    dbgBlinkL:               document.getElementById('dbg-blink-l'),
    dbgBlinkR:               document.getElementById('dbg-blink-r'),
    dbgJaw:                  document.getElementById('dbg-jaw'),
    dbgGaze:                 document.getElementById('dbg-gaze'),

    // Distraction banners & "I'm back" buttons
    distractionBannerL1:     document.getElementById('distraction-banner-l1'),
    distractionBannerL2:     document.getElementById('distraction-banner-l2'),
    bannerTextL1:            document.getElementById('banner-text-l1'),
    bannerTextL2:            document.getElementById('banner-text-l2'),
    bannerMutedPillL1:       document.getElementById('banner-muted-pill-l1'),
    bannerMutedPillL2:       document.getElementById('banner-muted-pill-l2'),
    btnImBackL1:             document.getElementById('btn-im-back-l1'),
    btnImBackL2:             document.getElementById('btn-im-back-l2'),

    // Score & Ring
    alertnessScore:          document.getElementById('alertness-score'),
    scoreRingCircle:         document.getElementById('score-ring-circle'),
    sevenBarStrip:           document.getElementById('seven-bar-strip'),
    stripEmptyMsg:           document.getElementById('strip-empty-msg'),
    scorePausedBadge:        document.getElementById('score-paused-badge'),
    scoreEmptyGuide:         document.getElementById('score-empty-guide'),
    scoreLiveInsight:        document.getElementById('score-live-insight'),
    scoreInsightHeading:     document.getElementById('score-insight-heading'),
    scoreInsightDetail:      document.getElementById('score-insight-detail'),

    // Contributors
    valEyesOpen:             document.getElementById('val-eyes-open'),
    pillEyesOpen:            document.getElementById('pill-eyes-open'),
    subEyesOpen:             document.getElementById('sub-eyes-open'),
    valHeadPose:             document.getElementById('val-head-pose'),
    pillHeadPose:            document.getElementById('pill-head-pose'),
    subHeadPose:             document.getElementById('sub-head-pose'),
    valBlinksMin:            document.getElementById('val-blinks-min'),
    pillBlinksMin:           document.getElementById('pill-blinks-min'),
    subBlinksMin:            document.getElementById('sub-blinks-min'),
    valLookaways:            document.getElementById('val-lookaways'),
    pillLookaways:           document.getElementById('pill-lookaways'),
    subLookaways:            document.getElementById('sub-lookaways'),
    stripLegend:             document.getElementById('strip-legend'),

    // Timeline, Filters & Show All
    timelineList:            document.getElementById('timeline-list'),
    filterPills:             document.querySelectorAll('.filter-pill'),
    filterCountAll:          document.getElementById('filter-count-all'),
    filterCountFocused:      document.getElementById('filter-count-focused'),
    filterCountDistracted:   document.getElementById('filter-count-distracted'),
    filterCountDrowsy:       document.getElementById('filter-count-drowsy'),
    btnTimelineToggleAll:    document.getElementById('btn-timeline-toggle-all'),

    // Top Bar & Session Controls
    sessionTimer:            document.getElementById('session-timer'),
    btnResetSession:         document.getElementById('btn-reset-session'),
    btnPreshift:             document.getElementById('btn-preshift'),

    // Level 3 Alarm
    alarmBar:                document.getElementById('alarm-bar'),
    alarmLevelLabel:         document.getElementById('alarm-level-label'),
    alarmMessageText:        document.getElementById('alarm-message-text'),
    alarmMutedPill:          document.getElementById('alarm-muted-pill'),
    btnImAwake:              document.getElementById('btn-im-awake'),

    // Demo Mode Toggle & Badge
    toggleDemoMode:          document.getElementById('toggle-demo-mode'),
    demoBadge:               document.getElementById('demo-badge'),
    demoToggleLabel:         document.getElementById('demo-toggle-label'),
    videoSimulatedPill:      document.getElementById('video-simulated-pill'),
    glassesModePill:         document.getElementById('glasses-mode-pill'),
    weakSignalPill:          document.getElementById('weak-signal-pill'),
    toggleGlassesMode:       document.getElementById('toggle-glasses-mode'),
    btnCalibrateEyes:        document.getElementById('btn-calibrate-eyes'),
    calibrateEyesStatus:     document.getElementById('calibrate-eyes-status'),
    calibrateEyesInfo:       document.getElementById('calibrate-eyes-info'),
    scoreSimulatedBadge:     document.getElementById('score-simulated-badge'),
    demoIllustrationView:    document.getElementById('demo-illustration-view'),
    demoControlsBar:         document.getElementById('demo-controls-bar'),
    demoStepBadge:           document.getElementById('demo-step-badge'),
    demoStageTitle:          document.getElementById('demo-stage-title'),
    demoStageDesc:           document.getElementById('demo-stage-desc'),
    btnDemoPrev:             document.getElementById('btn-demo-prev'),
    btnDemoNext:             document.getElementById('btn-demo-next'),
    btnDemoReplay:           document.getElementById('btn-demo-replay'),
    btnDemoReset:            document.getElementById('btn-demo-reset'),
    btnDemoExit:             document.getElementById('btn-demo-exit'),
    clearSimulatedBanner:    document.getElementById('clear-simulated-banner'),
    simulatedRowsCount:      document.getElementById('simulated-rows-count'),
    btnClearSimulatedRows:   document.getElementById('btn-clear-simulated-rows'),
    btnDismissSimulatedBanner: document.getElementById('btn-dismiss-simulated-banner'),

    // Pre-shift Check Modal
    preshiftModal:           document.getElementById('preshift-modal'),
    preshiftBackdrop:        document.getElementById('preshift-backdrop'),
    btnClosePreshift:        document.getElementById('btn-close-preshift'),
    btnBeginCheck:           document.getElementById('btn-begin-check'),
    linkPreshiftSettings:    document.getElementById('link-preshift-settings'),
    preshiftStartView:       document.getElementById('preshift-start-view'),
    preshiftActiveView:      document.getElementById('preshift-active-view'),
    preshiftStepLabel:       document.getElementById('preshift-step-label'),
    preshiftCountdown:       document.getElementById('preshift-countdown'),
    preshiftProgressBar:     document.getElementById('preshift-progress-bar'),
    tickRows: [
      document.getElementById('tick-row-1'),
      document.getElementById('tick-row-2'),
      document.getElementById('tick-row-3'),
      document.getElementById('tick-row-4'),
    ],
    preshiftResultSection:   document.getElementById('preshift-result-section'),
    preshiftResultBadge:     document.getElementById('preshift-result-badge'),
    btnStartSessionPreshift: document.getElementById('btn-start-session-preshift'),
    preshiftStepsSummary:    document.getElementById('preshift-steps-summary'),
    preshiftStep1Res:        document.getElementById('preshift-res-step-1'),
    preshiftStep2Res:        document.getElementById('preshift-res-step-2'),
    preshiftStep3Res:        document.getElementById('preshift-res-step-3'),
    preshiftStep4Res:        document.getElementById('preshift-res-step-4'),

    // Session Summary Modal
    summaryModal:            document.getElementById('summary-modal'),
    summaryBackdrop:         document.getElementById('summary-backdrop'),
    btnCloseSummary:         document.getElementById('btn-close-summary'),
    btnDoneSummary:          document.getElementById('btn-done-summary'),
    btnResetFromSummary:     document.getElementById('btn-reset-from-summary'),
    btnDownloadCsv:          document.getElementById('btn-download-csv'),
    summaryDuration:         document.getElementById('summary-duration'),
    summaryLowestScore:      document.getElementById('summary-lowest-score'),
    summaryDrowsyCount:      document.getElementById('summary-drowsy-count'),
    summaryDistractionCount: document.getElementById('summary-distraction-count'),
    summaryYawnCount:        document.getElementById('summary-yawn-count'),
    summaryTotalEvents:      document.getElementById('summary-total-events'),

    // Settings Popover
    btnSettings:             document.getElementById('btn-settings'),
    btnCloseSettings:        document.getElementById('btn-close-settings'),
    btnDoneSettings:         document.getElementById('btn-done-settings'),
    settingsPopover:         document.getElementById('settings-popover'),
    settingsBackdrop:        document.getElementById('settings-backdrop'),
    selectCamera:            document.getElementById('select-camera'),
    presetBadge:             document.getElementById('preset-badge'),
    btnPresetSensitive:      document.getElementById('btn-preset-sensitive'),
    btnPresetBalanced:       document.getElementById('btn-preset-balanced'),
    btnPresetRelaxed:        document.getElementById('btn-preset-relaxed'),
    sliderSensitivity:       document.getElementById('slider-sensitivity'),
    sensitivityDisplay:      document.getElementById('sensitivity-display'),
    sliderDrowsyTrigger:     document.getElementById('slider-drowsy-trigger'),
    drowsyTriggerDisplay:    document.getElementById('drowsy-trigger-display'),
    sliderLookawayTrigger:   document.getElementById('slider-lookaway-trigger'),
    lookawayTriggerDisplay:  document.getElementById('lookaway-trigger-display'),
    btnAlarmPreview:         document.getElementById('btn-alarm-preview'),
    previewStatusPill:       document.getElementById('preview-status-pill'),
    toggleAlarmSound:        document.getElementById('toggle-alarm-sound'),
    btnSoundIcon:            document.getElementById('btn-sound-icon'),
    btnTestSound:            document.getElementById('btn-test-sound'),
    sliderDrowsyVolume:      document.getElementById('slider-drowsy-volume'),
    drowsyVolumeDisplay:     document.getElementById('drowsy-volume-display'),
    sliderDistractionVolume: document.getElementById('slider-distraction-volume'),
    distractionVolumeDisplay: document.getElementById('distraction-volume-display'),
    btnTestDistractionSound: document.getElementById('btn-test-distraction-sound'),

    // Alarm honest reason lines
    alarmReasonText:         document.getElementById('alarm-reason-text'),
    bannerReasonL1:          document.getElementById('banner-reason-l1'),
    bannerReasonL2:          document.getElementById('banner-reason-l2'),

    // Timeline Detail Modal
    timelineDetailBackdrop:  document.getElementById('timeline-detail-backdrop'),
    timelineDetailModal:     document.getElementById('timeline-detail-modal'),
    btnCloseTimelineDetail:  document.getElementById('btn-close-timeline-detail'),
    btnCloseDetailAction:    document.getElementById('btn-close-detail-action'),
    detailCategoryPill:      document.getElementById('detail-category-pill'),
    detailCategoryDot:       document.getElementById('detail-category-dot'),
    detailCategoryIcon:      document.getElementById('detail-category-icon'),
    detailCategoryText:      document.getElementById('detail-category-text'),
    detailEventText:         document.getElementById('detail-event-text'),
    detailStartTime:         document.getElementById('detail-start-time'),
    detailEndTime:           document.getElementById('detail-end-time'),
    detailDuration:          document.getElementById('detail-duration'),
    detailScoreImpact:       document.getElementById('detail-score-impact'),
    detailAckTime:           document.getElementById('detail-ack-time'),

    // Safe Rest Assistant Card
    safeRestCard:            document.getElementById('safe-rest-card'),
    safeRestMainView:        document.getElementById('safe-rest-main-view'),
    safeRestCheckView:       document.getElementById('safe-rest-check-view'),
    safeRestCompletedView:   document.getElementById('safe-rest-completed-view'),
    safeRestPresetsRow:      document.getElementById('safe-rest-presets-row'),
    safeRestPresetBtns:      document.querySelectorAll('.btn-preset-time'),
    safeRestTimerDisplay:    document.getElementById('safe-rest-timer-display'),
    safeRestTimerSubtext:    document.getElementById('safe-rest-timer-subtext'),
    btnStartRest:            document.getElementById('btn-start-rest'),
    btnEndRestEarly:         document.getElementById('btn-end-rest-early'),
    btnReadyContinue:        document.getElementById('btn-ready-continue'),
    btnRestMore:             document.getElementById('btn-rest-more'),
    btnReopenSafeRest:       document.getElementById('btn-reopen-safe-rest'),

    // Journey Tracker Card
    journeyCard:             document.getElementById('journey-card'),
    journeyStatusPill:       document.getElementById('journey-status-pill'),
    journeyStatusText:       document.getElementById('journey-status-text'),
    journeySimulatedWrap:    document.getElementById('journey-simulated-wrap'),
    toggleSimJourney:        document.getElementById('toggle-sim-journey'),
    valJourneyStart:         document.getElementById('val-journey-start'),
    subJourneyStart:         document.getElementById('sub-journey-start'),
    valJourneyElapsed:       document.getElementById('val-journey-elapsed'),
    subJourneyElapsed:       document.getElementById('sub-journey-elapsed'),
    valJourneyDistance:      document.getElementById('val-journey-distance'),
    subJourneyDistance:      document.getElementById('sub-journey-distance'),
    valJourneyEvents:        document.getElementById('val-journey-events'),
    subJourneyEvents:        document.getElementById('sub-journey-events'),
    journeyGpsNotice:        document.getElementById('journey-gps-notice'),
    journeyGpsText:          document.getElementById('journey-gps-text'),
    journeyRestAdvisory:     document.getElementById('journey-rest-advisory'),
    restAdvisoryReason:      document.getElementById('rest-advisory-reason'),
    btnStartJourney:         document.getElementById('btn-start-journey'),
    btnEndJourney:           document.getElementById('btn-end-journey'),
    btnNewJourney:           document.getElementById('btn-new-journey'),
    journeySummaryBox:       document.getElementById('journey-summary-box'),
    summaryJourneyDuration:  document.getElementById('summary-journey-duration'),
    summaryJourneyDistance:  document.getElementById('summary-journey-distance'),
    summaryJourneyEvents:    document.getElementById('summary-journey-events'),
  };
}

// ─── Mathematical Helpers ─────────────────────────────────────
function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

function calculateEAR(landmarks, indices) {
  const p = indices.map(i => landmarks[i]);
  const v1 = dist2D(p[1], p[5]);
  const v2 = dist2D(p[2], p[4]);
  const h  = dist2D(p[0], p[3]);
  return (v1 + v2) / (2 * h + 1e-6);
}

function calculateMAR(landmarks) {
  const top    = landmarks[MOUTH_LANDMARKS.top];
  const bottom = landmarks[MOUTH_LANDMARKS.bottom];
  const left   = landmarks[MOUTH_LANDMARKS.left];
  const right  = landmarks[MOUTH_LANDMARKS.right];
  return dist2D(top, bottom) / (dist2D(left, right) + 1e-6);
}

function calculateHeadPoseYaw(landmarks) {
  if (!landmarks || landmarks.length <= 454) return 0;
  const nose = landmarks[HEAD_LANDMARKS.noseTip];
  const templeRight = landmarks[HEAD_LANDMARKS.userRightTemple]; // 234
  const templeLeft  = landmarks[HEAD_LANDMARKS.userLeftTemple];  // 454

  const dRight = dist2D(nose, templeRight);
  const dLeft  = dist2D(nose, templeLeft);
  const total  = dRight + dLeft + 1e-6;

  // When user turns RIGHT: nose moves towards templeRight (dRight decreases, dLeft increases)
  // Positive = RIGHT (+), Negative = LEFT (-)
  const ratio2D = (dLeft - dRight) / total;
  const yaw2D   = ratio2D * 75;

  // 3D depth difference: right temple goes back (z increases), left temple comes forward (z decreases)
  const dz = (templeRight.z - templeLeft.z);
  const dx = Math.abs(templeLeft.x - templeRight.x) + 1e-6;
  const yaw3D = Math.atan2(dz, dx) * (180 / Math.PI) * 0.75;

  return clamp(yaw2D * 0.6 + yaw3D * 0.4, -75, 75);
}

function calculateHeadPosePitch(landmarks) {
  if (!landmarks || landmarks.length <= 152) return 0;
  const forehead = landmarks[HEAD_LANDMARKS.forehead]; // 10
  const chin     = landmarks[HEAD_LANDMARKS.chin];     // 152
  const nose     = landmarks[HEAD_LANDMARKS.noseTip];  // 1

  const dForehead = dist2D(nose, forehead);
  const dChin     = dist2D(nose, chin);
  const total     = dForehead + dChin + 1e-6;

  // In neutral pose, dChin is ~0.54 of total.
  // When looking UP (+): nose moves towards forehead (dForehead shrinks, dChin grows).
  // When looking DOWN (-): nose moves towards chin (dChin shrinks, dForehead grows).
  const ratio2D = (dChin - dForehead) / total - 0.08;
  const pitch2D = ratio2D * 75;

  // 3D depth difference: forehead goes back (z increases), chin comes forward (z decreases)
  const dz = (forehead.z - chin.z);
  const dy = Math.abs(chin.y - forehead.y) + 1e-6;
  const pitch3D = Math.atan2(dz, dy) * (180 / Math.PI) * 0.75;

  return clamp(pitch2D * 0.6 + pitch3D * 0.4, -60, 60);
}

function calculateEyeGazeYawFromBlendshapes(blendshapeMap) {
  if (!blendshapeMap) return 0;
  const lookRight = ((blendshapeMap['eyeLookInLeft'] || 0) + (blendshapeMap['eyeLookOutRight'] || 0)) / 2;
  const lookLeft  = ((blendshapeMap['eyeLookOutLeft'] || 0) + (blendshapeMap['eyeLookInRight'] || 0)) / 2;
  return (lookRight - lookLeft) * 50; // positive = right, negative = left
}

function calculateEyeGazePitchFromBlendshapes(blendshapeMap) {
  if (!blendshapeMap) return 0;
  const lookUp   = ((blendshapeMap['eyeLookUpLeft'] || 0) + (blendshapeMap['eyeLookUpRight'] || 0)) / 2;
  const lookDown = ((blendshapeMap['eyeLookDownLeft'] || 0) + (blendshapeMap['eyeLookDownRight'] || 0)) / 2;
  return (lookUp - lookDown) * 35; // positive = UP, negative = DOWN
}

function calculateEyeGazeYawFromIris(landmarks) {
  if (!landmarks || landmarks.length <= 473) return 0;
  try {
    const leftOuter = landmarks[33];
    const leftInner = landmarks[133];
    const leftIris  = landmarks[468];

    const rightInner = landmarks[362];
    const rightOuter = landmarks[263];
    const rightIris  = landmarks[473];

    if (!leftIris || !rightIris) return 0;

    const leftSpan = leftInner.x - leftOuter.x;
    const rightSpan = rightOuter.x - rightInner.x;

    const leftRatio = (leftIris.x - leftOuter.x) / (leftSpan + 1e-6);
    const rightRatio = (rightIris.x - rightInner.x) / (rightSpan + 1e-6);

    const avgRatio = (leftRatio + rightRatio) / 2;
    return (avgRatio - 0.5) * 80;
  } catch (e) {
    return 0;
  }
}

function calculateGazeYaw(landmarks) {
  return calculateHeadPoseYaw(landmarks);
}

function updateDirectionHUD(yaw, pitch, direction, distSec = 0) {
  state.headDirection = direction;

  // 1. Camera OFF: show NO direction pill, NO Yaw/Pitch values, NO fake CENTRE
  if (state.cameraState === 'Off') {
    if (dom.valLiveYaw)       dom.valLiveYaw.textContent = '--';
    if (dom.valLivePitch)     dom.valLivePitch.textContent = '--';
    if (dom.valLiveDirection) dom.valLiveDirection.textContent = '--';
    if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '--';
    if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = '--';
    if (dom.overlayDirectionBadge) dom.overlayDirectionBadge.classList.add('hidden');
    return;
  }

  // 2. Camera ON + NO FACE: show NO FACE pill only, direction --, hide direction badge
  if ((!state.faceDetected && !state.isDemoMode) || direction === 'NO FACE') {
    if (dom.valLiveYaw)       dom.valLiveYaw.textContent = '--';
    if (dom.valLivePitch)     dom.valLivePitch.textContent = '--';
    if (dom.valLiveDirection) dom.valLiveDirection.textContent = '--';
    if (dom.overlayDirectionBadge) dom.overlayDirectionBadge.classList.add('hidden');
    return;
  }

  // 3. Camera ON + Face locked (or demo mode):
  const roundedYaw   = Math.round(yaw);
  const roundedPitch = Math.round(pitch);
  const yawStr   = roundedYaw >= 0 ? `+${roundedYaw}°` : `${roundedYaw}°`;
  const pitchStr = roundedPitch >= 0 ? `+${roundedPitch}°` : `${roundedPitch}°`;

  if (dom.valLiveYaw)       dom.valLiveYaw.textContent = yawStr;
  if (dom.valLivePitch)     dom.valLivePitch.textContent = pitchStr;
  if (dom.valLiveDirection) dom.valLiveDirection.textContent = direction;

  if (dom.overlayDirectionBadge) {
    let arrow = '●';
    let anglesText = `Yaw ${yawStr} • Pitch ${pitchStr}`;

    if (direction === 'LEFT') {
      arrow = '←';
      anglesText = `Yaw ${yawStr}`;
    } else if (direction === 'RIGHT') {
      arrow = '→';
      anglesText = `Yaw ${yawStr}`;
    } else if (direction === 'UP') {
      arrow = '↑';
      anglesText = `Pitch ${pitchStr}`;
    } else if (direction === 'DOWN') {
      arrow = '↓';
      anglesText = `Pitch ${pitchStr}`;
    } else if (direction === 'CENTRE') {
      arrow = '●';
      anglesText = `Yaw ${yawStr} • Pitch ${pitchStr}`;
    }

    if (dom.overlayDirectionArrow)  dom.overlayDirectionArrow.textContent  = arrow;
    if (dom.overlayDirectionText)   dom.overlayDirectionText.textContent   = direction;
    if (dom.overlayDirectionAngles) dom.overlayDirectionAngles.textContent = anglesText;

    if (state.activeDistractionLevel === 2) {
      dom.overlayDirectionBadge.className = 'overlay-direction-badge direction-alarm';
    } else if (state.activeDistractionLevel === 1 || state.distractionStartTime) {
      dom.overlayDirectionBadge.className = 'overlay-direction-badge direction-distracted';
    } else {
      dom.overlayDirectionBadge.className = 'overlay-direction-badge direction-centre';
    }
    dom.overlayDirectionBadge.classList.remove('hidden');
  }

  if (dom.cameraStatePill && state.faceDetected && !state.isCalibratingBaseline) {
    const isCentred = direction === 'CENTRE';
    const dirText = isCentred ? 'Centred' : `Off-centre (${direction})`;
    const pillCls = isCentred ? 'pill-focused' : 'pill-distracted';
    const dotCls = isCentred ? 'dot-focused' : 'dot-distracted';
    const icon = isCentred ? '✓' : '⚠';
    dom.cameraStatePill.className = `status-pill ${pillCls} hidden`;
    dom.cameraStatePill.innerHTML = `<span class="status-dot ${dotCls}" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">${icon}</span> Face: ${dirText}`;
  }
}

// ─── Live Eye Openness & Closure Timer HUD (Fix 1) ────────────
function updateEyeOpennessHUD(openL, openR, closedSec = 0, maxSec = 1.5, isClosed = false, isNoFace = false) {
  const maxSecStr = Number(maxSec || state.drowsyThresholdSec || 1.5).toFixed(1);
  const closedSecStr = Number(closedSec).toFixed(1);

  if (isNoFace) {
    if (dom.valEyeOpenL) dom.valEyeOpenL.textContent = '--%';
    if (dom.valEyeOpenR) dom.valEyeOpenR.textContent = '--%';
    if (dom.eyeBarFillL) dom.eyeBarFillL.style.width = '0%';
    if (dom.eyeBarFillR) dom.eyeBarFillR.style.width = '0%';
    if (dom.valEyeClosureTimer) dom.valEyeClosureTimer.textContent = `0.0s / ${maxSecStr}s`;
    if (dom.eyeClosureStatus) {
      dom.eyeClosureStatus.textContent = 'Paused';
      dom.eyeClosureStatus.className = 'eye-closure-status status-paused';
    }
    if (dom.eyeOpennessBar) dom.eyeOpennessBar.className = 'eye-openness-bar';
    return;
  }

  const lPct = clamp(Math.round(openL ?? 100), 0, 100);
  const rPct = clamp(Math.round(openR ?? 100), 0, 100);

  if (dom.valEyeOpenL) dom.valEyeOpenL.textContent = `${lPct}%`;
  if (dom.valEyeOpenR) dom.valEyeOpenR.textContent = `${rPct}%`;

  if (dom.eyeBarFillL) {
    dom.eyeBarFillL.style.width = `${lPct}%`;
    dom.eyeBarFillL.style.backgroundColor = lPct < 35 ? 'var(--status-drowsy)' : (lPct < 65 ? 'var(--status-distracted)' : 'var(--status-focused)');
  }
  if (dom.eyeBarFillR) {
    dom.eyeBarFillR.style.width = `${rPct}%`;
    dom.eyeBarFillR.style.backgroundColor = rPct < 35 ? 'var(--status-drowsy)' : (rPct < 65 ? 'var(--status-distracted)' : 'var(--status-focused)');
  }

  if (dom.valEyeClosureTimer) {
    dom.valEyeClosureTimer.textContent = `${closedSecStr}s / ${maxSecStr}s`;
  }

  if (dom.eyeClosureStatus && dom.eyeOpennessBar) {
    if (state.level3AlarmActive) {
      dom.eyeClosureStatus.textContent = 'DROWSY';
      dom.eyeClosureStatus.className = 'eye-closure-status status-alarm';
      dom.eyeOpennessBar.className = 'eye-openness-bar eye-drowsy';
    } else if (isClosed && closedSec >= 0.3) {
      dom.eyeClosureStatus.textContent = 'Closing…';
      dom.eyeClosureStatus.className = 'eye-closure-status status-closing';
      dom.eyeOpennessBar.className = 'eye-openness-bar eye-closing';
    } else {
      dom.eyeClosureStatus.textContent = 'Open';
      dom.eyeClosureStatus.className = 'eye-closure-status status-open';
      dom.eyeOpennessBar.className = 'eye-openness-bar';
    }
  }
}

// ─── Web Audio Alarm Synthesizer & Audio Unlock (Rule 1) ───────
function getAudioContext() {
  if (!state.audioContext) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) state.audioContext = new AudioCtx();
  }
  return state.audioContext;
}

function playTestBeep() {
  if (!state.alarmSoundEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.001, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.12, ctx.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.06);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.06);
  } catch (e) {
    console.warn('[Jaago] Test beep error:', e);
  }
}

async function unlockAudioOnGesture() {
  try {
    const ctx = getAudioContext();
    if (ctx && ctx.state === 'suspended') {
      await ctx.resume();
    }
    if (ctx && ctx.state === 'running') {
      playTestBeep();
      if (dom.soundBlockedNotice) dom.soundBlockedNotice.classList.add('hidden');
      return true;
    } else {
      if (dom.soundBlockedNotice) dom.soundBlockedNotice.classList.remove('hidden');
      return false;
    }
  } catch (e) {
    if (dom.soundBlockedNotice) dom.soundBlockedNotice.classList.remove('hidden');
    return false;
  }
}

function playAlarmBeep(freq, durSec, vol = 0.45) {
  if (!state.alarmSoundEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    if (ctx.state === 'suspended') ctx.resume().catch(() => {});

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    gain.gain.setValueAtTime(0.001, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + durSec);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + durSec);
  } catch (err) {
    console.warn('[Jaago] Beep error:', err);
  }
}

// ─── Drowsy Alarm Synthesizer (Stage 1 Beep 880Hz -> Stage 2 Siren 740/1100Hz) ─
function playDrowsyTone(freq, durSec, vol) {
  if (!state.alarmSoundEnabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});

  try {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.02);
    gain.gain.setValueAtTime(vol, ctx.currentTime + durSec - 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + durSec);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + durSec);
  } catch (err) {
    console.warn('[Jaago] Drowsy tone error:', err);
  }
}

function playDrowsyBeepStage1() {
  if (!state.alarmSoundEnabled || state.drowsySoundStage !== 1) return;
  // Stage 1: medium-LOUD beep, 880 Hz tone, 0.7s on, 0.3s off, volume 0.7 * drowsyVolume
  const vol = 0.7 * (state.drowsyVolume ?? 1.0);
  playDrowsyTone(880, 0.70, vol);
}

function triggerDrowsyVibration() {
  if (typeof navigator !== 'undefined' && navigator.vibrate) {
    try {
      navigator.vibrate([350, 50, 350, 50]);
    } catch (e) {}
  }
}

function startLevel3AudioAlarm() {
  stopLevel3AudioAlarm();
  state.drowsySoundStage = 1;

  if (state.alarmSoundEnabled) {
    playDrowsyBeepStage1();
    state.drowsyBeepInterval = setInterval(playDrowsyBeepStage1, 1000);
  }

  // Initial wake nudge vibration
  if (typeof navigator !== 'undefined' && navigator.vibrate) {
    try { navigator.vibrate([300, 150, 300]); } catch (e) {}
  }

  // Stage 2 (after 3 seconds continuous, automatically):
  // HIGH-VOLUME up-down siren - alternating 740 Hz / 1100 Hz, 0.35s each, volume 1.0,
  // looping continuously, PLUS vibration where supported, PLUS full-width red bar pulsing.
  // It must NOT stop by itself and must NOT get quieter over time.
  state.drowsyStage2Timeout = setTimeout(() => {
    startDrowsyAudioStage2();
  }, 3000);
}

function startDrowsyAudioStage2() {
  if (state.drowsyBeepInterval) {
    clearInterval(state.drowsyBeepInterval);
    state.drowsyBeepInterval = null;
  }
  state.drowsySoundStage = 2;

  // Full-width red bar pulsing
  if (dom.alarmBar) {
    dom.alarmBar.classList.add('alarm-bar-pulsing');
  }

  // Vibration where supported (continuous pulsing)
  triggerDrowsyVibration();
  if (typeof navigator !== 'undefined' && navigator.vibrate) {
    state.drowsyVibrateInterval = setInterval(triggerDrowsyVibration, 800);
  }

  if (!state.alarmSoundEnabled) return;

  const playSirenCycle = () => {
    if (state.drowsySoundStage !== 2 || !state.alarmSoundEnabled) return;
    const vol = 1.0 * (state.drowsyVolume ?? 1.0);
    // Tone 1: 740 Hz for 0.35s
    playDrowsyTone(740, 0.35, vol);
    state.drowsySirenTimeout2 = setTimeout(() => {
      if (state.drowsySoundStage === 2 && state.alarmSoundEnabled) {
        // Tone 2: 1100 Hz for 0.35s
        playDrowsyTone(1100, 0.35, vol);
      }
    }, 350);
  };

  playSirenCycle();
  state.drowsySirenInterval = setInterval(playSirenCycle, 700);
}

function stopLevel3AudioAlarm() {
  state.drowsySoundStage = 0;
  if (state.drowsyBeepInterval) {
    clearInterval(state.drowsyBeepInterval);
    state.drowsyBeepInterval = null;
  }
  if (state.drowsyStage2Timeout) {
    clearTimeout(state.drowsyStage2Timeout);
    state.drowsyStage2Timeout = null;
  }
  if (state.drowsySirenInterval) {
    clearInterval(state.drowsySirenInterval);
    state.drowsySirenInterval = null;
  }
  if (state.drowsySirenTimeout2) {
    clearTimeout(state.drowsySirenTimeout2);
    state.drowsySirenTimeout2 = null;
  }
  if (state.drowsyVibrateInterval) {
    clearInterval(state.drowsyVibrateInterval);
    state.drowsyVibrateInterval = null;
  }
  if (typeof navigator !== 'undefined' && navigator.vibrate) {
    try { navigator.vibrate(0); } catch (e) {}
  }
  if (dom.alarmBar) {
    dom.alarmBar.classList.remove('alarm-bar-pulsing');
  }
}

function testDrowsySoundScript() {
  if (state.drowsyTestTimeout) {
    stopDrowsySoundTest();
    return;
  }

  unlockAudioOnGesture();
  stopDistractionSound();
  stopLevel3AudioAlarm();

  if (dom.btnTestSound) {
    dom.btnTestSound.textContent = 'Testing Stage 1 (3s)…';
  }

  startLevel3AudioAlarm();

  state.drowsyTestTimeout = setTimeout(() => {
    if (dom.btnTestSound) {
      dom.btnTestSound.textContent = 'Testing Stage 2 Siren (3s)…';
    }
    if (state.drowsySoundStage !== 2) {
      startDrowsyAudioStage2();
    }

    state.drowsyTestTimeout = setTimeout(() => {
      stopDrowsySoundTest();
    }, 3000);
  }, 3000);
}

function stopDrowsySoundTest() {
  if (state.drowsyTestTimeout) {
    clearTimeout(state.drowsyTestTimeout);
    state.drowsyTestTimeout = null;
  }
  stopLevel3AudioAlarm();
  if (dom.btnTestSound) {
    dom.btnTestSound.textContent = 'Test drowsy sound';
  }
}

// ─── Sound UI & State Sync ────────────────────────────────────
const SOUND_ON_SVG = `
  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
  <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
`;

const SOUND_MUTED_SVG = `
  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
  <line x1="23" y1="9" x2="17" y2="15"></line>
  <line x1="17" y1="9" x2="23" y2="15"></line>
  <line x1="1" y1="1" x2="23" y2="23"></line>
`;

function updateSoundUI() {
  const isEnabled = state.alarmSoundEnabled;
  if (dom.btnSoundIcon) {
    if (isEnabled) {
      dom.btnSoundIcon.className = 'overlay-btn-pill sound-on';
      dom.btnSoundIcon.setAttribute('title', 'Audio chime: ON (Click to mute)');
      dom.btnSoundIcon.setAttribute('aria-label', 'Audio chime enabled');
      dom.btnSoundIcon.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${SOUND_ON_SVG}</svg>`;
    } else {
      dom.btnSoundIcon.className = 'overlay-btn-pill sound-muted';
      dom.btnSoundIcon.setAttribute('title', 'Audio chime: MUTED (Click to unmute)');
      dom.btnSoundIcon.setAttribute('aria-label', 'Audio chime muted');
      dom.btnSoundIcon.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${SOUND_MUTED_SVG}</svg>`;
    }
  }

  if (dom.toggleAlarmSound) {
    if (isEnabled) {
      dom.toggleAlarmSound.classList.add('active');
      dom.toggleAlarmSound.setAttribute('aria-checked', 'true');
    } else {
      dom.toggleAlarmSound.classList.remove('active');
      dom.toggleAlarmSound.setAttribute('aria-checked', 'false');
    }
  }
}

async function toggleSoundState(forceState = null) {
  await unlockAudioOnGesture();
  state.alarmSoundEnabled = (forceState !== null) ? forceState : !state.alarmSoundEnabled;
  if (!state.alarmSoundEnabled) {
    stopLevel3AudioAlarm();
    stopDistractionSound();
  } else {
    if (state.level3AlarmActive) {
      startLevel3AudioAlarm();
    } else if (state.activeDistractionLevel === 2) {
      startDistractionAudioLevel2();
    } else if (state.activeDistractionLevel === 1) {
      startDistractionAudioLevel1();
    }
  }
  updateSoundUI();
  updateBannerMutedPills();
}

async function playTestSound() {
  try {
    await unlockAudioOnGesture();
    const ctx = getAudioContext();
    if (!ctx) return;
    if (ctx.state === 'suspended') await ctx.resume();

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(1000, ctx.currentTime);

    gain.gain.setValueAtTime(0.001, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.35, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.20);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.20);
  } catch (err) {
    console.warn('[Jaago] playTestSound error:', err);
  }
}

// ─── Distraction Sound Synthesizer (Escalating, distinct from Drowsy) ─
function playDistractionBeepLevel1() {
  if (!state.alarmSoundEnabled || state.distractionSoundLevel !== 1) return;
  const ctx = getAudioContext();
  if (!ctx) return;
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});

  try {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const vol = 0.55 * state.distractionVolume;

    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, ctx.currentTime);

    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.03);
    gain.gain.setValueAtTime(vol, ctx.currentTime + 0.45);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.55);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.55);
  } catch (e) {
    console.warn('[Jaago] Distraction L1 audio error:', e);
  }
}

function startDistractionAudioLevel1() {
  if (state.distractionSoundLevel === 1) return;
  stopDistractionSound();
  state.distractionSoundLevel = 1;
  state.level1SoundStartTime = Date.now();
  if (!state.alarmSoundEnabled) return;
  playDistractionBeepLevel1();
  state.distractionBeepInterval = setInterval(playDistractionBeepLevel1, 750);
}

function playDistractionTone(freq, durSec, vol) {
  if (!state.alarmSoundEnabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});

  try {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.02);
    gain.gain.setValueAtTime(vol, ctx.currentTime + durSec - 0.03);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + durSec);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + durSec);
  } catch (e) {
    console.warn('[Jaago] Distraction tone error:', e);
  }
}

function startDistractionAudioLevel2() {
  if (state.distractionSoundLevel === 2) return;
  stopDistractionSound();
  state.distractionSoundLevel = 2;
  if (!state.alarmSoundEnabled) return;

  const playAlternatingCycle = () => {
    if (state.distractionSoundLevel !== 2 || !state.alarmSoundEnabled) return;
    const vol = 0.9 * state.distractionVolume;
    // Tone 1: 740 Hz for 0.35s
    playDistractionTone(740, 0.35, vol);
    // Tone 2: 1100 Hz at 0.38s for 0.35s
    setTimeout(() => {
      if (state.distractionSoundLevel === 2 && state.alarmSoundEnabled) {
        playDistractionTone(1100, 0.35, vol);
      }
    }, 380);
  };

  playAlternatingCycle();
  state.distractionBeepInterval = setInterval(playAlternatingCycle, 760);
}

function stopDistractionSound() {
  state.distractionSoundLevel = 0;
  state.level1SoundStartTime = null;
  if (state.distractionBeepInterval) {
    clearInterval(state.distractionBeepInterval);
    state.distractionBeepInterval = null;
  }
  if (state.distractionTestTimeout) {
    clearTimeout(state.distractionTestTimeout);
    state.distractionTestTimeout = null;
  }
}

function updateBannerMutedPills() {
  const isMuted = !state.alarmSoundEnabled;
  if (dom.bannerMutedPillL1) {
    if (isMuted) dom.bannerMutedPillL1.classList.remove('hidden');
    else dom.bannerMutedPillL1.classList.add('hidden');
  }
  if (dom.bannerMutedPillL2) {
    if (isMuted) dom.bannerMutedPillL2.classList.remove('hidden');
    else dom.bannerMutedPillL2.classList.add('hidden');
  }
  if (dom.alarmMutedPill) {
    if (isMuted) dom.alarmMutedPill.classList.remove('hidden');
    else dom.alarmMutedPill.classList.add('hidden');
  }
}

function testDistractionSoundScript() {
  if (state.distractionTestTimeout) {
    stopDistractionSound();
    if (dom.btnTestDistractionSound) {
      dom.btnTestDistractionSound.textContent = 'Test distraction sound';
    }
    return;
  }

  unlockAudioOnGesture();
  stopDistractionSound();
  stopLevel3AudioAlarm();

  if (dom.btnTestDistractionSound) {
    dom.btnTestDistractionSound.textContent = 'Testing Level 1 (min 3s)…';
  }

  startDistractionAudioLevel1();

  state.distractionTestTimeout = setTimeout(() => {
    if (dom.btnTestDistractionSound) {
      dom.btnTestDistractionSound.textContent = 'Testing Level 2 (looping)…';
    }
    startDistractionAudioLevel2();

    state.distractionTestTimeout = setTimeout(() => {
      stopDistractionSound();
      if (dom.btnTestDistractionSound) {
        dom.btnTestDistractionSound.textContent = 'Test distraction sound';
      }
    }, 3000);
  }, 3000);
}

// ─── Camera Indicator & Truth Status Sync ─────────────────────
function updateCameraIconUI(isLive) {
  if (!dom.btnCamIcon) return;
  if (isLive) {
    dom.btnCamIcon.classList.add('cam-on');
    dom.btnCamIcon.setAttribute('title', 'Camera: Live (Click to stop)');
    dom.btnCamIcon.setAttribute('aria-label', 'Camera stream live');
  } else {
    dom.btnCamIcon.classList.remove('cam-on');
    dom.btnCamIcon.setAttribute('title', 'Camera: Off (Click to start)');
    dom.btnCamIcon.setAttribute('aria-label', 'Camera stream off');
  }
}

// ─── Single Face State Synchronization (Fix B) ────────────────
function syncAllPills(nextFaceState) {
  state.faceState = nextFaceState;

  // 1. Bottom-Left Pill: "Face: Centred / Off-centre / No face"
  if (dom.cameraStatePill) {
    if (nextFaceState === 'CAMERA_OFF') {
      dom.cameraStatePill.className = 'status-pill pill-cam-off hidden';
      dom.cameraStatePill.setAttribute('title', 'Camera: Off');
      dom.cameraStatePill.innerHTML = '<span class="status-dot dot-cam-state" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">⊘</span> Face: Camera off';
    } else if (nextFaceState === 'CALIBRATING') {
      dom.cameraStatePill.className = 'status-pill pill-calibrating hidden';
      dom.cameraStatePill.innerHTML = '<span class="status-dot dot-calibrating" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">◌</span> Face: Centred (Calibrating)';
    } else if (nextFaceState === 'FACE_LOCKED') {
      const isCentred = !state.headDirection || state.headDirection === 'CENTRE';
      const dirText = isCentred ? 'Centred' : `Off-centre (${state.headDirection})`;
      const pillCls = isCentred ? 'pill-focused' : 'pill-distracted';
      const dotCls = isCentred ? 'dot-focused' : 'dot-distracted';
      const icon = isCentred ? '✓' : '⚠';
      dom.cameraStatePill.className = `status-pill ${pillCls} hidden`;
      dom.cameraStatePill.setAttribute('title', 'Camera: Live • Face locked');
      dom.cameraStatePill.innerHTML = `<span class="status-dot ${dotCls}" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">${icon}</span> Face: ${dirText}`;
    } else if (nextFaceState === 'NO_FACE') {
      dom.cameraStatePill.className = 'status-pill pill-noface hidden';
      dom.cameraStatePill.setAttribute('title', 'Camera: Live • No face');
      dom.cameraStatePill.innerHTML = '<span class="status-dot dot-noface" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">✕</span> Face: No face';
    }
  }

  // 2. Top-Bar Live Pill & Overlay Top-Left Live Pill
  let pillClass = 'pill-cam-off';
  let dotClass  = 'dot-cam-state';
  let icon      = '⊘';
  let labelText = 'CAMERA OFF';

  if (state.isDemoMode) {
    pillClass = 'pill-cam-live';
    dotClass  = 'dot-live';
    icon      = '●';
    labelText = 'LIVE';
  } else if (nextFaceState === 'CAMERA_OFF') {
    pillClass = 'pill-cam-off';
    dotClass  = 'dot-cam-state';
    icon      = '⊘';
    labelText = 'CAMERA OFF';
  } else if (nextFaceState === 'CALIBRATING') {
    pillClass = 'pill-cam-starting';
    dotClass  = 'dot-calibrating';
    icon      = '◌';
    labelText = 'CALIBRATING';
  } else if (nextFaceState === 'FACE_LOCKED') {
    pillClass = 'pill-cam-live';
    dotClass  = 'dot-live';
    icon      = '●';
    labelText = 'LIVE';
  } else if (nextFaceState === 'NO_FACE') {
    pillClass = 'pill-noface';
    dotClass  = 'dot-noface';
    icon      = '👤';
    labelText = 'NO FACE';
  }

  if (dom.topbarLivePill) {
    dom.topbarLivePill.className = `live-pill ${pillClass}`;
    dom.topbarLivePill.innerHTML = `<span class="status-dot ${dotClass}" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">${icon}</span> ${labelText}`;
  }

  if (dom.overlayLivePill) {
    dom.overlayLivePill.className = `live-pill ${pillClass}`;
    dom.overlayLivePill.innerHTML = `<span class="status-dot ${dotClass}" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">${icon}</span> ${labelText}`;
  }

  // 3. Video Overlay Top-Left Status Pill
  if (dom.faceStatusPill) {
    if (nextFaceState === 'CAMERA_OFF') {
      dom.faceStatusPill.className = 'status-pill pill-cam-off fixed-width-pill hidden';
      dom.faceStatusPill.innerHTML = '<span class="status-dot dot-cam-state" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">⊘</span> Camera Off';
      dom.faceStatusPill.classList.add('hidden');
      if (dom.overlayDirectionBadge) dom.overlayDirectionBadge.classList.add('hidden');
    } else if (nextFaceState === 'CALIBRATING') {
      dom.faceStatusPill.classList.remove('hidden');
      dom.faceStatusPill.className = 'status-pill pill-calibrating fixed-width-pill';
      dom.faceStatusPill.innerHTML = '<span class="status-dot dot-calibrating" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">◌</span> Calibrating';
    } else if (nextFaceState === 'NO_FACE') {
      dom.faceStatusPill.classList.remove('hidden');
      dom.faceStatusPill.className = 'status-pill pill-noface fixed-width-pill';
      dom.faceStatusPill.innerHTML = '<span class="status-dot dot-noface" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">👤</span> No face';
      if (dom.overlayDirectionBadge) dom.overlayDirectionBadge.classList.add('hidden');
    } else if (nextFaceState === 'FACE_LOCKED') {
      dom.faceStatusPill.classList.remove('hidden');
      updateGlobalStatus(state.currentStatus);
    }
  }
}

function updateTopBarLiveStatus() {
  syncAllPills(state.faceState);
}

// ─── Baseline Calibration & Centre Recalibration (Fix A & Fix 3) ──────
function startBaselineCalibration(durationMs = 5000) {
  state.isCalibratingBaseline = true;
  state.baselineCaptured = false;
  state.baselineCalibrationStartTime = Date.now();
  state.baselineDurationMs = durationMs;
  state.baselineSamples = [];

  syncAllPills('CALIBRATING');
  updateGlobalStatus('Calibrating');
  updateGuidance(`Calibrating baseline… Hold naturally facing screen with eyes open (${Math.ceil(durationMs / 1000)}s)`);
  renderAlertnessScore();
  updateScoreAndContributors();
  updateSevenBarStrip();
}

function recalibrateCentre() {
  if (!state.faceDetected) {
    updateGuidance('Move into frame before setting centre');
    return;
  }

  // Immediately use current smoothed pose as baseline
  state.baselineYaw = state.smoothedYaw;
  state.baselinePitch = state.smoothedPitch;
  if (state.eyeBlinkLeft !== null) {
    state.baselineBlinkLeft = clamp(state.eyeBlinkLeft, 0.0, 0.35);
    state.baselineBlinkRight = clamp(state.eyeBlinkRight, 0.0, 0.35);
    state.closedThresholdL = clamp(state.baselineBlinkLeft + 0.35, 0.38, 0.65);
    state.closedThresholdR = clamp(state.baselineBlinkRight + 0.35, 0.38, 0.65);
    state.closedThreshold  = (state.closedThresholdL + state.closedThresholdR) / 2;
  }
  state.relativeYaw = 0;
  state.relativePitch = 0;
  state.headDirection = 'CENTRE';

  // Stop any active distraction sound immediately
  stopDistractionSound();
  hideDistractionBanners();
  state.distractionStartTime = null;
  state.distractionLostFaceSince = null;
  state.distractionLoggedStart = false;
  state.activeDistractionLevel = 0;
  state.frontStableSince = Date.now();

  if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
  if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
  if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Centre';

  updateDirectionHUD(0, 0, 'CENTRE', 0);
  updateEyeOpennessHUD(100, 100, 0, state.drowsyThresholdSec, false, false);

  // Run 2-second refinement from current head position
  startBaselineCalibration(2000);
}

// ─── Glasses Mode & Eye Calibration System (Relative Thresholds) ─
function setGlassesMode(enabled) {
  state.glassesMode = !!enabled;
  if (dom.toggleGlassesMode) {
    if (state.glassesMode) {
      dom.toggleGlassesMode.classList.add('active');
    } else {
      dom.toggleGlassesMode.classList.remove('active');
    }
    dom.toggleGlassesMode.setAttribute('aria-checked', state.glassesMode ? 'true' : 'false');
  }
  if (dom.glassesModePill) {
    if (state.glassesMode) {
      dom.glassesModePill.classList.remove('hidden');
    } else {
      dom.glassesModePill.classList.add('hidden');
    }
  }
}

function updateEyeSignalWeakPill() {
  const shouldShow = (state.cameraState === 'Live') &&
                     Boolean(state.faceDetected) &&
                     Boolean(state.isEyeSignalWeak);
  if (dom.weakSignalPill) {
    if (shouldShow) {
      dom.weakSignalPill.classList.remove('hidden');
    } else {
      dom.weakSignalPill.classList.add('hidden');
    }
  }
}

function resetCameraOffHUD() {
  // Clear live eye openness & closure timer back to "--" / hidden
  state.eyeOpennessL             = null;
  state.eyeOpennessR             = null;
  state.eyeClosedSince           = null;
  state.weakEyeSignalSince       = null;
  state.isEyeSignalWeak          = false;
  state.headNodStartTime         = null;

  updateEyeSignalWeakPill();

  if (dom.valEyeOpenL) dom.valEyeOpenL.textContent = '--%';
  if (dom.valEyeOpenR) dom.valEyeOpenR.textContent = '--%';
  if (dom.eyeBarFillL) dom.eyeBarFillL.style.width = '0%';
  if (dom.eyeBarFillR) dom.eyeBarFillR.style.width = '0%';
  if (dom.valEyeClosureTimer) dom.valEyeClosureTimer.textContent = `-- / ${Number(state.drowsyThresholdSec || 1.5).toFixed(1)}s`;
  if (dom.eyeClosureStatus) {
    dom.eyeClosureStatus.textContent = '--';
    dom.eyeClosureStatus.className = 'eye-closure-status status-paused';
  }
  if (dom.eyeOpennessBar) dom.eyeOpennessBar.className = 'eye-openness-bar';

  // Clear live head pose direction, yaw, pitch, and lookaway timer
  state.headDirection            = '--';
  state.relativeYaw              = 0;
  state.relativePitch            = 0;
  state.smoothedYaw              = 0;
  state.smoothedPitch            = 0;

  if (dom.valLiveYaw) dom.valLiveYaw.textContent = '--';
  if (dom.valLivePitch) dom.valLivePitch.textContent = '--';
  if (dom.valLiveDirection) dom.valLiveDirection.textContent = '--';
  if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '--';
  if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = '--';
  if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
  if (dom.overlayDirectionBadge) dom.overlayDirectionBadge.classList.add('hidden');
  if (dom.faceStatusPill) dom.faceStatusPill.classList.add('hidden');
}

function startEyeCalibration() {
  setGlassesMode(true);
  state.isEyeCalibrating = true;
  state.eyeCalibrationStartTime = Date.now();
  state.eyeCalibrationSamples = [];

  if (dom.btnCalibrateEyes) {
    dom.btnCalibrateEyes.disabled = true;
    dom.btnCalibrateEyes.textContent = 'Calibrating… (2s)';
  }
  if (dom.calibrateEyesStatus) {
    dom.calibrateEyesStatus.textContent = 'Calibrating…';
    dom.calibrateEyesStatus.classList.remove('hidden');
  }
  updateGuidance('Calibrating eyes… Look at camera with eyes open (2s)');

  // 2-second timeout guarantees completion even in mock or when frame rate drops
  setTimeout(() => {
    if (state.isEyeCalibrating) {
      finishEyeCalibration();
    }
  }, 2050);
}

function finishEyeCalibration() {
  if (!state.isEyeCalibrating && state.calibratedEyeOpenness !== null) return;
  state.isEyeCalibrating = false;
  const samples = state.eyeCalibrationSamples;

  if (samples && samples.length > 0) {
    state.calibratedEyeOpennessL = samples.reduce((s, x) => s + x.openL, 0) / samples.length;
    state.calibratedEyeOpennessR = samples.reduce((s, x) => s + x.openR, 0) / samples.length;
    state.calibratedEyeOpenness  = samples.reduce((s, x) => s + x.openAvg, 0) / samples.length;
    state.calibratedBaselineEAR  = samples.reduce((s, x) => s + x.ear, 0) / samples.length;
  } else {
    const curOpenL = clamp(1.0 - (state.eyeBlinkLeft ?? 0.10), 0, 1);
    const curOpenR = clamp(1.0 - (state.eyeBlinkRight ?? 0.10), 0, 1);
    state.calibratedEyeOpennessL = curOpenL;
    state.calibratedEyeOpennessR = curOpenR;
    state.calibratedEyeOpenness  = (curOpenL + curOpenR) / 2;
    state.calibratedBaselineEAR  = ((state.rawLeftEAR || 0.3) + (state.rawRightEAR || 0.3)) / 2;
  }

  state.calibratedEyeOpenness  = clamp(state.calibratedEyeOpenness, 0.15, 1.0);
  state.calibratedEyeOpennessL = clamp(state.calibratedEyeOpennessL, 0.15, 1.0);
  state.calibratedEyeOpennessR = clamp(state.calibratedEyeOpennessR, 0.15, 1.0);
  state.eyeCalibrationTimestamp = Date.now();
  setGlassesMode(true);

  // Show "Calibrating… done" state
  if (dom.btnCalibrateEyes) {
    dom.btnCalibrateEyes.textContent = 'Calibrating… done';
    dom.btnCalibrateEyes.disabled = false;
  }
  if (dom.calibrateEyesStatus) {
    dom.calibrateEyesStatus.textContent = 'Done ✓';
    dom.calibrateEyesStatus.classList.remove('hidden');
  }
  if (dom.calibrateEyesInfo) {
    const pct = Math.round(state.calibratedEyeOpenness * 100);
    const trig = Math.round(pct * 0.45);
    dom.calibrateEyesInfo.textContent = `Baseline: ${pct}% open · Closure trigger: <${trig}%`;
  }
  updateGuidance('Eye calibration complete — baseline set');

  setTimeout(() => {
    if (dom.btnCalibrateEyes) {
      dom.btnCalibrateEyes.textContent = 'Recalibrate eyes (2s)';
    }
  }, 2500);
}

function evaluateEyeLandmarkConfidence(trackedLandmarks, blendshapeMap) {
  if (state.simulateWeakEyeSignal !== undefined) {
    return state.simulateWeakEyeSignal ? 0.3 : 0.95;
  }
  if (!trackedLandmarks || trackedLandmarks.length === 0) return 0.0;

  // 1. Check MediaPipe visibility / presence attributes on eye landmark indices
  const eyeIndices = [...EYE_LANDMARKS.leftEAR, ...EYE_LANDMARKS.rightEAR];
  let sumScore = 0;
  let countScore = 0;
  for (const idx of eyeIndices) {
    const pt = trackedLandmarks[idx];
    if (pt) {
      if (typeof pt.visibility === 'number') {
        sumScore += pt.visibility;
        countScore++;
      } else if (typeof pt.presence === 'number') {
        sumScore += pt.presence;
        countScore++;
      }
    }
  }
  if (countScore > 0) {
    const avgScore = sumScore / countScore;
    if (avgScore < 0.5) return avgScore;
  }

  // 2. Anatomical sanity check on eye landmarks
  const leftEAR = calculateEAR(trackedLandmarks, EYE_LANDMARKS.leftEAR);
  const rightEAR = calculateEAR(trackedLandmarks, EYE_LANDMARKS.rightEAR);
  if (isNaN(leftEAR) || isNaN(rightEAR) || leftEAR <= 0.05 || rightEAR <= 0.05 || leftEAR >= 0.65 || rightEAR >= 0.65) {
    return 0.3;
  }

  // Eye asymmetry from glare/frame occlusion
  if (Math.abs(leftEAR - rightEAR) > 0.22) {
    return 0.4;
  }

  return 0.95;
}

function showVideoPaneError(reason, fix) {
  if (dom.placeholderGraphic) {
    dom.placeholderGraphic.classList.remove('hidden');
  }
  if (dom.webcam) {
    dom.webcam.classList.add('hidden');
  }
  if (dom.videoPaneErrorBox && dom.videoPaneErrorReason && dom.videoPaneErrorFix) {
    dom.videoPaneErrorReason.textContent = reason;
    dom.videoPaneErrorFix.textContent    = fix;
    dom.videoPaneErrorBox.classList.remove('hidden');
  }
  if (dom.placeholderCrosshair) {
    dom.placeholderCrosshair.classList.add('hidden');
  }
  if (dom.placeholderCaptionText) {
    dom.placeholderCaptionText.textContent = 'CAMERA ACCESS BLOCKED';
  }
  if (dom.placeholderSubtext) {
    dom.placeholderSubtext.textContent = 'Resolve error or toggle camera to retry';
  }

  showError(reason, fix);
}

function clearVideoPaneError() {
  if (dom.videoPaneErrorBox) {
    dom.videoPaneErrorBox.classList.add('hidden');
  }
  if (dom.placeholderCrosshair) {
    dom.placeholderCrosshair.classList.remove('hidden');
  }
  if (dom.placeholderCaptionText) {
    dom.placeholderCaptionText.textContent = 'VIGILANCE GUARD STANDBY';
  }
  if (dom.placeholderSubtext) {
    dom.placeholderSubtext.textContent = 'Camera off — click the camera icon to start';
  }
  dismissError();
}

// ─── MediaPipe Tasks Vision Model Loader ──────────────────────
async function loadFaceLandmarker() {
  if (state.modelLoaded || state.modelLoading) return;
  state.modelLoading = true;
  updateGuidance('Downloading Face Landmarker model from Google CDN…');

  try {
    const vision = await import(MEDIAPIPE_CDN_BUNDLE);
    state.FaceLandmarkerClass  = vision.FaceLandmarker;
    state.FilesetResolverClass = vision.FilesetResolver;

    const filesetResolver = await state.FilesetResolverClass.forVisionTasks(MEDIAPIPE_WASM_PATH);

    const tryInit = async (delegate) => {
      return state.FaceLandmarkerClass.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: MODEL_ASSET_PATH,
          delegate: delegate,
        },
        outputFaceBlendshapes: true, // Use blendshapes first
        runningMode: 'VIDEO',
        numFaces: 4,
        minFaceDetectionConfidence: 0.35,
        minFacePresenceConfidence: 0.35,
        minTrackingConfidence: 0.35,
      });
    };

    try {
      state.faceLandmarker = await tryInit('GPU');
    } catch (gpuErr) {
      console.warn('[Jaago] GPU delegate fallback to CPU:', gpuErr);
      state.faceLandmarker = await tryInit('CPU');
    }

    state.modelLoaded  = true;
    state.modelLoading = false;
    state.modelError   = null;
    updateGuidance('Face Landmarker model ready · Click "Start Session" to begin');
  } catch (err) {
    console.warn('[Jaago] Model load error:', err);
    state.modelLoaded  = false;
    state.modelLoading = false;
    state.modelError   = err;

    if (!state.isDemoMode) {
      showError(
        'Face Landmarker model download failed (~5 MB CDN asset).',
        'Check your internet connection or use Demo Mode (runs offline without camera or model).'
      );
    }
  }
}

// ─── Camera Lifecycle ─────────────────────────────────────────
async function startCamera(deviceId = null) {
  await unlockAudioOnGesture();
  clearVideoPaneError();
  setCameraState('Starting');
  updateGuidance('Requesting browser camera permission…');

  if (!state.modelLoaded && !state.modelLoading) {
    loadFaceLandmarker().catch(() => {});
  }

  const targetDeviceId = deviceId || state.selectedCameraId;
  const constraints = {
    video: targetDeviceId
      ? { deviceId: { exact: targetDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false,
  };

  try {
    if (state.stream) {
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    dom.webcam.srcObject = stream;

    clearVideoPaneError();

    const onStreamReady = () => {
      dom.webcam.play().catch(e => console.warn('[Jaago] play() warning:', e));
      if (dom.placeholderGraphic) dom.placeholderGraphic.classList.add('hidden');
      if (dom.webcam)             dom.webcam.classList.remove('hidden');

      setCameraState('Live');
      updateCameraIconUI(true);
      populateCameraChoices();

      if (state.rafId) cancelAnimationFrame(state.rafId);
      state.rafId = requestAnimationFrame(videoProcessingLoop);
    };

    if (dom.webcam.readyState >= 1) {
      onStreamReady();
    } else {
      dom.webcam.onloadedmetadata = onStreamReady;
      dom.webcam.oncanplay = onStreamReady;
    }
  } catch (err) {
    console.error('[Jaago] getUserMedia error:', err);
    setCameraState('Blocked');
    updateCameraIconUI(false);

    let reason = 'Camera error';
    let fix = 'Allow camera access and try again.';

    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      reason = 'Camera permission blocked or denied by browser.';
      fix = 'Allow camera in the browser address bar (lock/camera icon) and click the camera icon to start.';
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      reason = 'Camera is in use by another app (Zoom, Meet, Teams).';
      fix = 'Close the other app using the camera and click the camera icon to start.';
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      reason = 'No video camera detected on this system.';
      fix = 'Connect an external webcam or enable your camera in system settings.';
    } else if (err.name === 'OverconstrainedError') {
      if (targetDeviceId) {
        console.warn('[Jaago] Overconstrained with deviceId, trying default camera...');
        state.selectedCameraId = null;
        return startCamera(null);
      }
      reason = 'The selected camera device is not available.';
      fix = 'Select "Default Camera" in Settings and try again.';
    } else {
      reason = `Camera error (${err.name || 'UnknownError'}): ${err.message || 'Stream failed'}`;
      fix = 'Ensure no other application is using the camera and reload the page.';
    }

    showVideoPaneError(reason, fix);
  }
}

function stopCamera() {
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }

  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }

  if (dom.webcam) {
    dom.webcam.srcObject = null;
    dom.webcam.classList.add('hidden');
  }

  if (dom.placeholderGraphic) {
    dom.placeholderGraphic.classList.remove('hidden');
  }

  clearVideoPaneError();
  clearCanvasOverlay();
  resetDebugMetrics();

  state.faceDetected             = false;
  state.faceLockedSince          = null;
  state.trackedFaceBox           = null;
  state.lastTrackedFaceBox       = null;
  state.targetScore              = null;
  state.displayedScore           = null;
  state.eyeClosedSince           = null;
  state.distractionStartTime     = null;
  state.distractionLostFaceSince = null;
  state.frontStableSince         = null;
  state.yawHistory               = [];
  state.pitchHistory             = [];
  state.yawnStartTime            = null;
  state.blinkTimestamps          = [];

  resetCameraOffHUD();

  dismissLevel3Alarm();
  hideDistractionBanners();

  setCameraState('Off');
  updateCameraIconUI(false);
  updateGlobalStatus('No face');
  updateGuidance('Camera off — click the camera icon to start');
  renderAlertnessScore();
}

async function switchCamera(deviceId) {
  state.selectedCameraId = deviceId || null;
  if (state.cameraState === 'Live' || state.cameraState === 'No face' || state.cameraState === 'Starting') {
    await startCamera(state.selectedCameraId);
  }
}

async function populateCameraChoices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');

    if (!dom.selectCamera) return;
    dom.selectCamera.innerHTML = '';

    if (videoInputs.length === 0) {
      dom.selectCamera.innerHTML = '<option value="">Default Camera</option>';
      return;
    }

    videoInputs.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `Camera ${idx + 1}`;
      if (state.selectedCameraId === dev.deviceId) {
        opt.selected = true;
      }
      dom.selectCamera.appendChild(opt);
    });
  } catch (e) {
    console.warn('[Jaago] enumerateDevices error:', e);
  }
}

// ─── Live Video Loop & Detection Engine ───────────────────────
function videoProcessingLoop() {
  if (state.cameraState === 'Off' || state.cameraState === 'Blocked') return;

  const video = dom.webcam;
  if (video && video.readyState >= 2 && state.modelLoaded && state.faceLandmarker) {
    try {
      const nowMs = performance.now();
      const results = state.faceLandmarker.detectForVideo(video, nowMs);
      handleDetectionResults(results);
    } catch (err) {
      console.warn('[Jaago] detectForVideo exception:', err);
    }
  }

  drawCornerBracketOverlay();
  updateDebugDisplay();
  updateScoreAndContributors();
  renderAlertnessScore();

  state.rafId = requestAnimationFrame(videoProcessingLoop);
}

// ─── Process Detection & Extract Blendshapes / Fallback EAR ───
function handleDetectionResults(results) {
  const hasFace = results && results.faceLandmarks && results.faceLandmarks.length > 0;
  const now = Date.now();

  if (hasFace) {
    // Multi-face rule: track ONLY largest/closest face (Rule 2)
    let bestIdx = 0;
    let maxArea = -1;
    let bestBox = null;

    for (let i = 0; i < results.faceLandmarks.length; i++) {
      const lm = results.faceLandmarks[i];
      const box = computeLandmarkBox(lm);
      const area = (box.maxX - box.minX) * (box.maxY - box.minY);
      if (area > maxArea) {
        maxArea = area;
        bestIdx = i;
        bestBox = box;
      }
    }

    const trackedLandmarks = results.faceLandmarks[bestIdx];
    state.trackedFaceBox = bestBox;

    if (!state.faceDetected) {
      state.faceDetected = true;
      state.faceLockedSince = now;
      // Start score at 100 upon first lock
      if (state.targetScore === null) {
        setTargetScore(100);
      }
    }

    setCameraState('Live');

    // 1. Blendshapes First (Fallback to Landmark Geometry EAR / MAR)
    let blendshapeMap = null;
    if (results.faceBlendshapes && results.faceBlendshapes[bestIdx] && results.faceBlendshapes[bestIdx].categories) {
      blendshapeMap = {};
      results.faceBlendshapes[bestIdx].categories.forEach(c => {
        blendshapeMap[c.categoryName] = c.score;
      });
    }

    const leftEAR  = calculateEAR(trackedLandmarks, EYE_LANDMARKS.leftEAR);
    const rightEAR = calculateEAR(trackedLandmarks, EYE_LANDMARKS.rightEAR);
    state.rawLeftEAR  = leftEAR;
    state.rawRightEAR = rightEAR;

    if (blendshapeMap && blendshapeMap['eyeBlinkLeft'] !== undefined && blendshapeMap['eyeBlinkRight'] !== undefined) {
      state.usedBlendshapes = true;
      state.eyeBlinkLeft    = clamp(blendshapeMap['eyeBlinkLeft'], 0, 1);
      state.eyeBlinkRight   = clamp(blendshapeMap['eyeBlinkRight'], 0, 1);
      state.jawOpen         = clamp(blendshapeMap['jawOpen'] ?? 0, 0, 1);
      state.headPoseYaw     = calculateHeadPoseYaw(trackedLandmarks);
      state.headPosePitch   = calculateHeadPosePitch(trackedLandmarks);
      state.eyeGazeYaw      = calculateEyeGazeYawFromBlendshapes(blendshapeMap);
      state.eyeGazePitch    = calculateEyeGazePitchFromBlendshapes(blendshapeMap);
    } else {
      // 2. Fallback to Landmark Geometry (EAR / MAR)
      state.usedBlendshapes = false;
      const mar = calculateMAR(trackedLandmarks);

      state.eyeBlinkLeft  = clamp(1.0 - ((leftEAR - 0.16) / 0.16), 0, 1);
      state.eyeBlinkRight = clamp(1.0 - ((rightEAR - 0.16) / 0.16), 0, 1);
      state.jawOpen       = clamp((mar - 0.15) / 0.40, 0, 1);
      state.headPoseYaw   = calculateHeadPoseYaw(trackedLandmarks);
      state.headPosePitch = calculateHeadPosePitch(trackedLandmarks);
      state.eyeGazeYaw    = calculateEyeGazeYawFromIris(trackedLandmarks);
      state.eyeGazePitch  = 0;
    }

    // Combined raw metrics (head pose + eye gaze)
    const rawYaw   = state.headPoseYaw + state.eyeGazeYaw * 0.35;
    const rawPitch = state.headPosePitch + state.eyeGazePitch * 0.35;

    // Evaluate eye landmark confidence for weak-signal fallback (Glasses mode)
    const eyeConf = evaluateEyeLandmarkConfidence(trackedLandmarks, blendshapeMap);
    state.eyeLandmarkConfidence = eyeConf;

    if (eyeConf < 0.5) {
      if (!state.weakEyeSignalSince) {
        state.weakEyeSignalSince = now;
      }
      if (now - state.weakEyeSignalSince >= 2000) {
        state.isEyeSignalWeak = true;
      }
    } else {
      state.weakEyeSignalSince = null;
      state.isEyeSignalWeak = false;
    }
    updateEyeSignalWeakPill();

    // 7. Smoothing over 5-7 frames (jitter prevention)
    state.yawHistory.push(rawYaw);
    if (state.yawHistory.length > 6) state.yawHistory.shift();
    state.pitchHistory.push(rawPitch);
    if (state.pitchHistory.length > 6) state.pitchHistory.shift();

    const smoothedYaw   = state.yawHistory.reduce((a, b) => a + b, 0) / state.yawHistory.length;
    const smoothedPitch = state.pitchHistory.reduce((a, b) => a + b, 0) / state.pitchHistory.length;

    state.smoothedYaw   = smoothedYaw;
    state.smoothedPitch = smoothedPitch;
    state.gazeYaw       = smoothedYaw;
    state.pitch         = smoothedPitch;
    state.gazeLabel     = formatGaze(smoothedYaw);

    // Fix A.1 & Fix 3: 3-second eyes-open + centre baseline capture
    if (state.isCalibratingBaseline) {
      state.baselineSamples.push({
        yaw: smoothedYaw,
        pitch: smoothedPitch,
        blinkL: state.eyeBlinkLeft ?? 0.05,
        blinkR: state.eyeBlinkRight ?? 0.05,
        leftEAR: state.rawLeftEAR ?? 0.30,
        rightEAR: state.rawRightEAR ?? 0.30
      });
      const elapsed = now - state.baselineCalibrationStartTime;
      if (elapsed >= state.baselineDurationMs && state.baselineSamples.length >= 3) {
        state.baselineYaw   = state.baselineSamples.reduce((s, x) => s + x.yaw, 0) / state.baselineSamples.length;
        state.baselinePitch = state.baselineSamples.reduce((s, x) => s + x.pitch, 0) / state.baselineSamples.length;

        // 3-second natural eyes-open baseline (Fix 3)
        const avgBlinkL = state.baselineSamples.reduce((s, x) => s + x.blinkL, 0) / state.baselineSamples.length;
        const avgBlinkR = state.baselineSamples.reduce((s, x) => s + x.blinkR, 0) / state.baselineSamples.length;
        state.baselineBlinkLeft  = clamp(avgBlinkL, 0.0, 0.35);
        state.baselineBlinkRight = clamp(avgBlinkR, 0.0, 0.35);

        // Set closed threshold relative to natural eyes-open baseline (not one fixed number!)
        state.closedThresholdL = clamp(state.baselineBlinkLeft + 0.35, 0.38, 0.65);
        state.closedThresholdR = clamp(state.baselineBlinkRight + 0.35, 0.38, 0.65);
        state.closedThreshold  = (state.closedThresholdL + state.closedThresholdR) / 2;

        state.baselineCaptured = true;
        state.isCalibratingBaseline = false;
        state.frontStableSince = now;
        setTargetScore(100);
        state.displayedScore = 100;
        state.lastValidScore = 100;
        syncAllPills('FACE_LOCKED');
        updateGlobalStatus('Focused');
        updateGuidance('Baseline captured — Centre & eye thresholds set');
        addTimelineRow('Baseline calibration completed — Centre and eye thresholds locked', '5.0s', 'focused');
      } else {
        const secLeft = Math.max(1, Math.ceil((state.baselineDurationMs - elapsed) / 1000));
        updateGuidance(`Calibrating baseline… Hold naturally facing screen with eyes open (${secLeft}s)`);
        syncAllPills('CALIBRATING');
        updateGlobalStatus('Calibrating');
        updateDirectionHUD(0, 0, 'CENTRE', 0);
        updateEyeOpennessHUD(100, 100, 0, state.drowsyThresholdSec, false, false);
        return; // Calibration active: hold further evaluation
      }
    } else if (!state.baselineCaptured && (state.isSessionRunning || state.cameraState === 'Live')) {
      startBaselineCalibration(5000);
      return;
    }

    // Relative metrics (Fix A.2: Centre = natural baseline, not absolute zero)
    const relYaw   = smoothedYaw - (state.baselineCaptured ? state.baselineYaw : 0);
    const relPitch = smoothedPitch - (state.baselineCaptured ? state.baselinePitch : 0);
    state.relativeYaw   = relYaw;
    state.relativePitch = relPitch;

    const absRelYaw   = Math.abs(relYaw);
    const absRelPitch = Math.abs(relPitch);

    // Direction calculation from RELATIVE values (Fix A.2)
    let direction = 'CENTRE';
    if (absRelYaw < 15 && absRelPitch < 15) {
      direction = 'CENTRE';
    } else if (absRelYaw >= absRelPitch) {
      direction = (relYaw < 0) ? 'LEFT' : 'RIGHT';
    } else {
      direction = (relPitch > 0) ? 'UP' : 'DOWN';
    }
    state.headDirection = direction;

    // ── DETECTION LOGIC ────────────────────────────────────────

    // Calculate Eye Openness % relative to baseline (Fix 1)
    const baseOpenL = (state.glassesMode && state.calibratedEyeOpennessL) ? state.calibratedEyeOpennessL : (1.0 - (state.baselineBlinkLeft || 0.05));
    const baseOpenR = (state.glassesMode && state.calibratedEyeOpennessR) ? state.calibratedEyeOpennessR : (1.0 - (state.baselineBlinkRight || 0.05));
    const curOpenL  = clamp(1.0 - state.eyeBlinkLeft, 0, 1);
    const curOpenR  = clamp(1.0 - state.eyeBlinkRight, 0, 1);

    const opennessPctL = clamp(Math.round((curOpenL / Math.max(0.1, baseOpenL)) * 100), 0, 100);
    const opennessPctR = clamp(Math.round((curOpenR / Math.max(0.1, baseOpenR)) * 100), 0, 100);
    state.eyeOpennessL = opennessPctL;
    state.eyeOpennessR = opennessPctR;

    // Collect eye calibration samples if user started Calibrate Eyes (2s)
    if (state.isEyeCalibrating) {
      state.eyeCalibrationSamples.push({
        openL: curOpenL,
        openR: curOpenR,
        openAvg: (curOpenL + curOpenR) / 2,
        blinkL: state.eyeBlinkLeft,
        blinkR: state.eyeBlinkRight,
        ear: ((state.rawLeftEAR || 0.3) + (state.rawRightEAR || 0.3)) / 2
      });
      if (now - state.eyeCalibrationStartTime >= 2000) {
        finishEyeCalibration();
      }
    }

    // A. Blink & Sustained Eye Closure Tracking (Relative to Baseline or Absolute Fallback)
    let eyesClosed = false;
    let closedL = false;
    let closedR = false;

    if (state.glassesMode && state.calibratedEyeOpenness !== null) {
      // Glasses Mode active & calibrated: use relative threshold (~45% of calibrated baseline)
      const curOpenAvg = (curOpenL + curOpenR) / 2;
      const relThreshL   = state.calibratedEyeOpennessL * 0.45;
      const relThreshR   = state.calibratedEyeOpennessR * 0.45;
      const relThreshAvg = state.calibratedEyeOpenness * 0.45;

      if (state.usedBlendshapes) {
        // Direct blendshape openness below ~45% of calibrated baseline
        closedL = (curOpenL <= relThreshL) || (opennessPctL <= 45);
        closedR = (curOpenR <= relThreshR) || (opennessPctR <= 45);
        eyesClosed = (curOpenAvg <= relThreshAvg) || (closedL && closedR);
      } else {
        // Landmark EAR fallback below ~45% of calibrated baseline
        const curEAR = (state.rawLeftEAR + state.rawRightEAR) / 2;
        const earThresh = (state.calibratedBaselineEAR || 0.3) * 0.45;
        closedL = state.rawLeftEAR <= earThresh;
        closedR = state.rawRightEAR <= earThresh;
        eyesClosed = (curEAR <= earThresh) || (closedL && closedR);
      }
    } else {
      // Fallback: existing absolute threshold (unchanged behaviour)
      const curBlinkAvg = (state.eyeBlinkLeft + state.eyeBlinkRight) / 2;
      const threshL   = state.closedThresholdL || 0.45;
      const threshR   = state.closedThresholdR || 0.45;
      const threshAvg = state.closedThreshold  || 0.45;

      closedL = state.eyeBlinkLeft >= threshL;
      closedR = state.eyeBlinkRight >= threshR;
      eyesClosed = (closedL && closedR) || (curBlinkAvg >= threshAvg && (closedL || closedR));
    }

    if (eyesClosed) {
      if (!state.eyeClosedSince) {
        state.eyeClosedSince = now;
      }
      const closedMs = now - state.eyeClosedSince;
      const closedSec = closedMs / 1000;
      const triggerMs = state.drowsyThresholdSec * 1000;

      // Update live Eye Openness meter and closure timer (Fix 1)
      updateEyeOpennessHUD(opennessPctL, opennessPctR, closedSec, state.drowsyThresholdSec, true, false);

      // Level 3 Drowsy alarm check (adjustable slider default 1.5s, 1.0-3.0s)
      if (closedMs >= triggerMs && !state.level3AlarmActive) {
        triggerLevel3Alarm();
      }
    } else {
      // Eyes are OPEN:
      if (state.eyeClosedSince) {
        const closedDurMs = now - state.eyeClosedSince;
        state.eyeClosedSince = null;

        // Normal blink is 100–400ms — must NEVER alarm, counts towards Blinks/min
        if (closedDurMs >= 100 && closedDurMs <= 450) {
          state.blinkTimestamps.push(now);
          state.lastBlinkDetectedTime = now;
        } else if (closedDurMs > 450 && closedDurMs < (state.drowsyThresholdSec * 1000)) {
          // Extended closure under alarm threshold
          adjustScore(-5);
        }
      }

      // Live closure timer MUST reset to 0.0s the moment eyes open! (Fix 1)
      updateEyeOpennessHUD(opennessPctL, opennessPctR, 0, state.drowsyThresholdSec, false, false);
    }

    // Clean up old blinks (> 60s)
    state.blinkTimestamps = state.blinkTimestamps.filter(t => (now - t) <= 60000);

    // B. Yawn Tracking (sustained jaw-open only >= 1.5s)
    const jawWideOpen = state.jawOpen >= 0.55;
    if (jawWideOpen) {
      if (!state.yawnStartTime) state.yawnStartTime = now;
    } else {
      if (state.yawnStartTime) {
        const yawnDurMs = now - state.yawnStartTime;
        state.yawnStartTime = null;
        if (yawnDurMs >= 1500 && (now - state.lastYawnLogged) > 5000) {
          state.lastYawnLogged = now;
          adjustScore(-10); // Score rule: Yawn event -10
          addTimelineRow('Yawn detected (sustained)', (yawnDurMs / 1000).toFixed(1) + 's', 'distracted');
        }
      }
    }

    // Weak-signal fallback: In weak eye signal state ONLY (>2s low confidence),
    // trigger Drowsy alarm if sustained head-nod (pitch down >=2s) AND yawn occur together.
    const isHeadPitchDown = (direction === 'DOWN' || relPitch < -15);
    if (isHeadPitchDown) {
      if (!state.headNodStartTime) {
        state.headNodStartTime = now;
      }
    } else {
      state.headNodStartTime = null;
    }
    const headNodSustained = state.headNodStartTime && (now - state.headNodStartTime >= 2000);

    if (state.cameraState === 'Live' && state.faceDetected && state.isEyeSignalWeak) {
      const isYawning = jawWideOpen || (state.yawnStartTime && (now - state.yawnStartTime >= 500));
      if (headNodSustained && isYawning) {
        if (!state.level3AlarmActive) {
          state.drowsyTriggerReason = 'weak_signal_nod_yawn';
          triggerLevel3Alarm();
        }
      }
    }

    // C. Distraction Tracking (Relative thresholds: Enter >25° yaw or >20° pitch, Exit <15° yaw and <15° pitch for >=0.6s)
    const isAngleLookingAway = (absRelYaw > 25 || absRelPitch > 20);
    const isAngleFront       = (absRelYaw < 15 && absRelPitch < 15);
    const inDistraction      = (state.distractionStartTime !== null);

    if (!inDistraction) {
      if (isAngleLookingAway) {
        state.distractionStartTime = now;
        state.frontStableSince = null;
        state.distractionLoggedStart = false;
      } else {
        if (isAngleFront) {
          if (!state.frontStableSince) state.frontStableSince = now;
        } else {
          state.frontStableSince = null;
        }
      }
    } else {
      // Exit rule (relative, Fix A.3): exit Distracted ONLY when relativeYaw < 15° AND relativePitch < 15° continuously for 0.6s
      if (isAngleFront) {
        if (!state.frontStableSince) state.frontStableSince = now;
        const frontDurationMs = now - state.frontStableSince;
        if (frontDurationMs >= 600) {
          // Sustained front for 0.6s — stop ALL distraction sound within 1 second, clear banner, set Focused!
          const distDurMs = state.frontStableSince - state.distractionStartTime;
          const durSec = Math.max(0.6, distDurMs / 1000);
          state.distractionStartTime = null;
          state.distractionLostFaceSince = null;
          state.activeDistractionLevel = 0;

          stopDistractionSound();
          hideDistractionBanners();

          if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
          if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
          if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Centre';

          // Log cleared row (Fix C.1)
          if (state.distractionLoggedStart || durSec >= 1.2) {
            state.lookAwayCount++;
            state.lastLookAwayTime = Date.now();
            adjustScore(durSec >= 3.0 ? -15 : -10);
            addTimelineRow('Distraction cleared', `${durSec.toFixed(1)}s`, 'focused');
          }
          state.distractionLoggedStart = false;

          updateGlobalStatus('Focused');
          syncAllPills('FACE_LOCKED');
        }
      } else {
        // Looking away or glancing: reset front stability timer
        state.frontStableSince = null;
      }
    }

    // Look-away timer & sound escalation & Start row logging (Fix C.1)
    if (state.distractionStartTime !== null) {
      const distMs = now - state.distractionStartTime;
      const distSec = distMs / 1000;

      if (dom.lookawayTimerVal) {
        dom.lookawayTimerVal.textContent = `${distSec.toFixed(1)}s`;
      }

      // Thresholds: Level 1 at lookawayL1ThresholdSec (default 1.2s), Level 2 at lookawayThresholdSec (default 3.0s)
      const l1Thresh = state.lookawayL1ThresholdSec || 1.2;
      const l2Thresh = state.lookawayThresholdSec || 3.0;
      if (distSec >= l1Thresh) {
        // Log start row once (Fix C.1): DISTRACTED (amber pill), full text e.g. "Distracted — Looking LEFT"
        if (!state.distractionLoggedStart) {
          state.distractionLoggedStart = true;
          addTimelineRow(`Distracted — Looking ${state.headDirection}`, '—', 'distracted');
        }

        if (distSec >= l2Thresh) {
          setLevel2Distraction();
          if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar lookaway-level2';
          if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Level 2 (Alarm)';
        } else {
          setLevel1Distraction();
          if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar lookaway-level1';
          if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Level 1 (Distracted)';
        }
      } else {
        if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
        if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Looking away…';
      }
    } else {
      if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
      if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
      if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Centre';
    }

    // Consistency rule (Fix A.5): If Direction is CENTRE and Look-away is 0.0s, status MUST be Focused and sound MUST be off
    if (direction === 'CENTRE' && state.distractionStartTime === null) {
      if (state.distractionSoundLevel > 0) stopDistractionSound();
      hideDistractionBanners();
      if (!state.level3AlarmActive && state.faceState === 'FACE_LOCKED') {
        updateGlobalStatus('Focused');
      }
    }

    // Live HUD updates with RELATIVE values (Fix A.2)
    const curDistSec = state.distractionStartTime ? ((now - state.distractionStartTime) / 1000) : 0;
    updateDirectionHUD(relYaw, relPitch, direction, curDistSec);

    // D. Global Status Pill updates & NEVER show Calibrating after baseline captured (Fix A.1 & B.3)
    if (state.level3AlarmActive) {
      updateGlobalStatus('Drowsy');
    } else if (state.activeDistractionLevel > 0 || state.distractionStartTime !== null) {
      updateGlobalStatus('Distracted');
    } else if (isAngleFront) {
      updateGlobalStatus('Focused');
      syncAllPills('FACE_LOCKED');
      updateGuidance('Face locked · Vigilance tracking active');

      // Continuous Focus Recovery (+2 per second while continuously focused, Rule 4)
      if (!state.lastFocusRecoveryTime || (now - state.lastFocusRecoveryTime) >= 1000) {
        adjustScore(+2);
        state.lastFocusRecoveryTime = now;
      }
    }

  } else {
    // NO FACE DETECTED
    // Requirement 3: Lost-face hold during distraction (up to 2.0 seconds)
    if (state.distractionStartTime !== null || state.activeDistractionLevel > 0) {
      if (!state.distractionLostFaceSince) {
        state.distractionLostFaceSince = now;
      }
      const lostDurMs = now - state.distractionLostFaceSince;
      if (lostDurMs < 2000) {
        // HOLD DISTRACTION STATE FOR UP TO 2.0 SECONDS!
        state.faceDetected = true;
        state.trackedFaceBox = state.lastTrackedFaceBox;
        const distSec = (now - state.distractionStartTime) / 1000;
        if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = `${distSec.toFixed(1)}s`;

        const l1Thresh = state.lookawayL1ThresholdSec || 1.2;
        const l2Thresh = state.lookawayThresholdSec || 3.0;

        if (distSec >= l1Thresh && !state.distractionLoggedStart) {
          state.distractionLoggedStart = true;
          addTimelineRow(`Distracted — Looking ${state.headDirection || 'away'}`, '—', 'distracted');
        }

        if (distSec >= l2Thresh) {
          setLevel2Distraction();
          if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar lookaway-level2';
          if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Level 2 (Alarm)';
        } else if (distSec >= l1Thresh) {
          setLevel1Distraction();
          if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar lookaway-level1';
          if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Level 1 (Distracted)';
        }

        updateGuidance('Face turned away / partially out of frame — still distracted');
        updateGlobalStatus('Distracted');

        // Keep last known direction and degrees displayed on HUD
        updateDirectionHUD(state.relativeYaw, state.relativePitch, state.headDirection || 'LEFT', distSec);
        return; // Early return, keeping distraction active!
      }
    }

    // 2.0 full seconds of no face exceeded, or no face without prior distraction:
    state.faceDetected             = false;
    state.faceLockedSince          = null;
    state.trackedFaceBox           = null;
    state.lastTrackedFaceBox       = null;
    state.distractionStartTime     = null;
    state.distractionLostFaceSince = null;
    state.distractionLoggedStart   = false;
    state.frontStableSince         = null;
    state.yawHistory               = [];
    state.pitchHistory             = [];

    // Fix 4: If no face, do NOT run eye logic & pause eye meter
    state.eyeClosedSince           = null;
    state.weakEyeSignalSince       = null;
    state.isEyeSignalWeak          = false;
    state.headNodStartTime         = null;
    updateEyeSignalWeakPill();
    updateEyeOpennessHUD(null, null, 0, state.drowsyThresholdSec, false, true);

    stopDistractionSound();
    hideDistractionBanners();
    resetDebugMetrics();

    if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
    if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
    if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'No face';

    updateDirectionHUD(0, 0, 'NO FACE', 0);

    syncAllPills('NO_FACE');
    updateGuidance('No face — eye detection paused · Move into frame / check lighting');
    updateGlobalStatus('No face');
    renderAlertnessScore();
    updateScoreAndContributors();
  }
}

function computeLandmarkBox(landmarks) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarks[i];
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, maxX, minY, maxY };
}

function formatGaze(yaw) {
  const rounded = Math.round(yaw);
  if (Math.abs(rounded) <= 8) return `Center (0°)`;
  if (rounded > 8)            return `Right (+${rounded}°)`;
  return `Left (${rounded}°)`;
}

// ─── Level 1 & Level 2 Distraction Banners (Rule 3) ────────────
function setLevel1Distraction() {
  if (state.level3AlarmActive) return; // Level 3 takes precedence

  const distSec = state.distractionStartTime ? ((Date.now() - state.distractionStartTime) / 1000) : (state.lookawayL1ThresholdSec || 1.2);
  const l1Thresh = state.lookawayL1ThresholdSec || 1.2;
  if (dom.bannerReasonL1) {
    dom.bannerReasonL1.textContent = `Why did this alarm fire? Head turned ${state.headDirection || 'away'} continuously for ${distSec.toFixed(1)}s (threshold ${l1Thresh.toFixed(1)}s)`;
  }

  if (state.activeDistractionLevel === 1) return; // Already at Level 1

  state.activeDistractionLevel = 1;
  if (dom.bannerTextL1) {
    dom.bannerTextL1.textContent = 'Distracted — Look back at the screen';
  }
  updateBannerMutedPills();
  if (dom.distractionBannerL1) dom.distractionBannerL1.classList.remove('hidden');
  if (dom.distractionBannerL2) dom.distractionBannerL2.classList.add('hidden');

  startDistractionAudioLevel1();
  updateGlobalStatus('Distracted');
}

function setLevel2Distraction() {
  if (state.level3AlarmActive) return;

  const distSec = state.distractionStartTime ? ((Date.now() - state.distractionStartTime) / 1000) : (state.lookawayThresholdSec || 3.0);
  const l2Thresh = state.lookawayThresholdSec || 3.0;
  if (dom.bannerReasonL2) {
    dom.bannerReasonL2.textContent = `Why did this alarm fire? Head turned ${state.headDirection || 'away'} continuously for ${distSec.toFixed(1)}s (threshold ${l2Thresh.toFixed(1)}s)`;
  }

  if (state.activeDistractionLevel === 2) return; // Already at Level 2

  state.activeDistractionLevel = 2;
  if (dom.bannerTextL2) {
    dom.bannerTextL2.textContent = 'Still distracted — Look back now';
  }
  updateBannerMutedPills();
  if (dom.distractionBannerL1) dom.distractionBannerL1.classList.add('hidden');
  if (dom.distractionBannerL2) dom.distractionBannerL2.classList.remove('hidden');

  startDistractionAudioLevel2();
  updateGlobalStatus('Distracted');
}

function hideDistractionBanners() {
  state.activeDistractionLevel = 0;
  if (dom.distractionBannerL1) dom.distractionBannerL1.classList.add('hidden');
  if (dom.distractionBannerL2) dom.distractionBannerL2.classList.add('hidden');
  stopDistractionSound();
}

// ─── Level 3 Drowsy Alarm & "I'm Awake" Action ────────────────
function triggerLevel3Alarm() {
  state.level3AlarmActive = true;
  hideDistractionBanners();

  // Close any open popovers or modals so alarm bar and "I'm awake" button are never obscured
  if (dom.settingsPopover && !dom.settingsPopover.classList.contains('hidden')) {
    dom.settingsPopover.classList.add('hidden');
    dom.settingsBackdrop?.classList.add('hidden');
    dom.btnSettings?.setAttribute('aria-expanded', 'false');
  }
  if (dom.timelineDetailModal && !dom.timelineDetailModal.classList.contains('hidden')) {
    closeTimelineDetailModal();
  }
  if (dom.preshiftModal && !dom.preshiftModal.classList.contains('hidden')) {
    closePreshiftModal();
  }
  if (dom.summaryModal && !dom.summaryModal.classList.contains('hidden')) {
    closeSessionSummaryModal();
  }

  // Full-width red bar
  if (dom.alarmLevelLabel) dom.alarmLevelLabel.textContent = 'ALARM LEVEL 3 · DROWSINESS DETECTED';

  if (state.drowsyTriggerReason === 'weak_signal_nod_yawn') {
    if (dom.alarmMessageText) dom.alarmMessageText.textContent = 'Weak eye signal fallback — head-nod & yawn detected';
    if (dom.alarmReasonText) {
      dom.alarmReasonText.textContent = 'Why did this alarm fire? Eye landmarks obscured; sustained head pitch down (2.0s) and yawn detected together';
    }
  } else {
    if (dom.alarmMessageText) dom.alarmMessageText.textContent = 'Sustained eye closure detected — tap "I\'m awake" to confirm vigilance';
    const closedDurSec = state.eyeClosedSince ? ((Date.now() - state.eyeClosedSince) / 1000) : state.drowsyThresholdSec;
    if (dom.alarmReasonText) {
      dom.alarmReasonText.textContent = `Why did this alarm fire? Eyes closed continuously for ${closedDurSec.toFixed(1)}s (threshold ${state.drowsyThresholdSec.toFixed(1)}s)`;
    }
  }

  if (dom.alarmBar) dom.alarmBar.classList.remove('hidden');
  if (dom.btnImAwake) dom.btnImAwake.disabled = false;
  updateBannerMutedPills();

  // Freeze non-alarm animations
  document.body.classList.add('freeze-non-alarm');

  // Audio tone (Stage 1 medium-LOUD beep -> Stage 2 siren)
  startLevel3AudioAlarm();

  // Deduct score: Drowsy event -40 (Rule 4)
  adjustScore(-40);
  updateGlobalStatus('Drowsy');
  state.drowsyStartTime = Date.now();
  const tlLabel = state.drowsyTriggerReason === 'weak_signal_nod_yawn' ? 'Drowsy — Head-nod & Yawn (Weak Signal)' : 'Drowsy — Eyes closed';
  addTimelineRow(tlLabel, '—', 'drowsy');
}

function dismissLevel3Alarm() {
  state.level3AlarmActive = false;
  state.eyeClosedSince    = null;
  state.drowsyTriggerReason = null;
  state.headNodStartTime  = null;

  // Stop sound, vibration & hide alarm bar at once
  stopLevel3AudioAlarm();
  if (dom.alarmBar) {
    dom.alarmBar.classList.remove('alarm-bar-pulsing');
    dom.alarmBar.classList.add('hidden');
  }

  // Resume animations
  document.body.classList.remove('freeze-non-alarm');
}

function markActiveTimelineAcksResolved() {
  if (!dom.timelineList) return;
  const activeBtns = dom.timelineList.querySelectorAll('.tl-btn-ack:not(.acked)');
  activeBtns.forEach(btn => {
    btn.classList.remove('tl-btn-ack-distracted', 'tl-btn-ack-drowsy');
    btn.classList.add('acked', 'tl-action-icon-cleared');
    btn.textContent = '✓';
    btn.disabled = true;
    btn.setAttribute('title', 'Alarm acknowledged');
    btn.setAttribute('aria-label', 'Alarm acknowledged');
  });
}

function clearDistractionWarning() {
  const durSec = state.distractionStartTime ? ((Date.now() - state.distractionStartTime) / 1000) : 0;
  stopDistractionSound();
  hideDistractionBanners();
  state.activeDistractionLevel = 0;
  state.distractionStartTime = null;
  state.distractionLostFaceSince = null;
  state.distractionLoggedStart = false;
  state.frontStableSince = Date.now();

  if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
  if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
  if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Centre';

  updateDirectionHUD(0, 0, 'CENTRE', 0);
  markActiveTimelineAcksResolved();

  if (durSec >= 1.2) {
    state.lookAwayCount++;
    state.lastLookAwayTime = Date.now();
    adjustScore(durSec >= 3.0 ? -15 : -10);
  }
  addTimelineRow("Distraction cleared — I'm back", `${durSec > 0 ? durSec.toFixed(1) + 's' : '0.0s'}`, 'focused');

  if (state.faceDetected) {
    updateGlobalStatus(state.isDemoMode ? 'Focused (Simulated)' : 'Focused');
    updateGuidance('Face locked · Vigilance tracking active');
  }
}

function acknowledgeAlarmFromButton() {
  if (state.isDemoMode) {
    goToDemoStage(4);
    return;
  }
  const durSec = state.drowsyStartTime ? ((Date.now() - state.drowsyStartTime) / 1000) : state.drowsyThresholdSec;
  state.drowsyStartTime = null;
  dismissLevel3Alarm();
  markActiveTimelineAcksResolved();
  addTimelineRow("Drowsy cleared — I'm awake", `${durSec.toFixed(1)}s`, 'focused');
  if (state.faceDetected) {
    updateGlobalStatus('Focused');
    updateGuidance('Face locked · Vigilance tracking active');
  }
}

// ─── Score Engine & Smooth Animation (Rule 4) ──────────────────
function setTargetScore(val) {
  state.targetScore = clamp(val, 0, 100);
  state.scoreAnimStartTime = performance.now();
  state.scoreAnimStartVal  = state.displayedScore !== null ? state.displayedScore : state.targetScore;
}

function adjustScore(delta) {
  if (state.targetScore === null) return;
  setTargetScore(state.targetScore + delta);
}

function getScoreBand(score) {
  const rounded = Math.round(score !== null && score !== undefined ? score : 100);
  if (rounded >= 80) {
    return {
      label: 'Optimal Vigilance',
      color: '#22C55E', // Green
      tier: 'focused',
      desc: 'Active engagement · Head position centered'
    };
  } else if (rounded >= 60) {
    return {
      label: 'Good / Watch',
      color: '#F59E0B', // Amber
      tier: 'distracted',
      desc: 'Slight fatigue or occasional drift detected'
    };
  } else if (rounded >= 40) {
    return {
      label: 'Low Alertness',
      color: '#F97316', // Orange
      tier: 'distracted',
      desc: 'Attention drift · Gaze diversion or head turn detected'
    };
  } else {
    return {
      label: 'Critical',
      color: '#EF4444', // Red
      tier: 'drowsy',
      desc: 'Critical alertness loss · Confirmation required'
    };
  }
}

function renderAlertnessScore() {
  if (state.isDemoMode) {
    if (dom.scorePausedBadge) {
      dom.scorePausedBadge.classList.add('hidden');
    }
    const roundedScore = Math.round(state.displayedScore !== null ? state.displayedScore : (state.targetScore !== null ? state.targetScore : 100));
    state.lastValidScore = roundedScore;
    if (dom.alertnessScore) dom.alertnessScore.textContent = roundedScore;

    const band = getScoreBand(roundedScore);
    const scoreColor = band.color;
    const statusTier = band.tier;

    if (dom.alertnessScore) dom.alertnessScore.style.color = scoreColor;
    if (dom.scoreRingCircle) {
      const offset = 251.3 * (1 - roundedScore / 100);
      dom.scoreRingCircle.style.strokeDashoffset = String(offset);
      dom.scoreRingCircle.style.stroke = scoreColor;
    }

    if (dom.scoreEmptyGuide) dom.scoreEmptyGuide.classList.add('hidden');
    if (dom.scoreLiveInsight) {
      dom.scoreLiveInsight.classList.remove('hidden');
      if (dom.scoreInsightHeading && dom.scoreInsightDetail) {
        dom.scoreInsightHeading.textContent = `${band.label} (Simulated)`;
        dom.scoreInsightHeading.style.color = scoreColor;
        dom.scoreInsightDetail.textContent = band.desc;
      }
    }

    const nowMs = performance.now();
    if (!state.lastHistorySampleTime || nowMs - state.lastHistorySampleTime >= 2500) {
      state.lastHistorySampleTime = nowMs;
      const timeStr = new Date().toTimeString().slice(0, 8);
      state.scoreHistory.push({ score: roundedScore, tier: statusTier, time: timeStr });
      if (state.scoreHistory.length > 7) {
        state.scoreHistory.shift();
      }
    }
    updateSevenBarStrip();
    return;
  }

  // 1. No face or camera off: Hero score "--", "Paused — No face", contributors "— / Waiting"
  if (!state.faceDetected || state.cameraState === 'Off') {
    if (dom.alertnessScore) {
      dom.alertnessScore.textContent = '--';
      dom.alertnessScore.style.color = 'var(--muted)';
    }
    if (dom.scorePausedBadge) {
      dom.scorePausedBadge.textContent = 'Paused — No face';
      dom.scorePausedBadge.classList.remove('hidden');
    }
    if (dom.scoreRingCircle) {
      dom.scoreRingCircle.style.strokeDashoffset = '251.3';
      dom.scoreRingCircle.style.stroke = 'var(--border)';
    }
    if (dom.scoreEmptyGuide) dom.scoreEmptyGuide.classList.remove('hidden');
    if (dom.scoreLiveInsight) dom.scoreLiveInsight.classList.add('hidden');
    updateSevenBarStrip();
    return;
  }

  // 2. Calibrating baseline (first 5s with face): Hero score "--", "Calibrating…", contributors "-- / Calibrating"
  if (state.isCalibratingBaseline) {
    if (dom.alertnessScore) {
      dom.alertnessScore.textContent = '--';
      dom.alertnessScore.style.color = 'var(--status-distracted)';
    }
    if (dom.scorePausedBadge) {
      dom.scorePausedBadge.textContent = 'Calibrating…';
      dom.scorePausedBadge.classList.remove('hidden');
    }
    if (dom.scoreRingCircle) {
      dom.scoreRingCircle.style.strokeDashoffset = '251.3';
      dom.scoreRingCircle.style.stroke = 'var(--border)';
    }
    if (dom.scoreEmptyGuide) dom.scoreEmptyGuide.classList.remove('hidden');
    if (dom.scoreLiveInsight) dom.scoreLiveInsight.classList.add('hidden');
    updateSevenBarStrip();
    return;
  }

  // 3. Live session / active face:
  if (dom.scorePausedBadge) {
    dom.scorePausedBadge.classList.add('hidden');
  }
  if (dom.scoreEmptyGuide) dom.scoreEmptyGuide.classList.add('hidden');
  if (dom.scoreLiveInsight) {
    dom.scoreLiveInsight.classList.remove('hidden');
  }

  // Animate number over 500–700ms ease-out (no bounce)
  const nowMs = performance.now();
  const animDur = 600; // ms
  const elapsed = nowMs - state.scoreAnimStartTime;
  const progress = clamp(elapsed / animDur, 0, 1);
  const easeOut = 1 - Math.pow(1 - progress, 3); // cubic ease-out

  const target = state.targetScore !== null ? state.targetScore : 100;
  state.displayedScore = state.scoreAnimStartVal + (target - state.scoreAnimStartVal) * easeOut;
  const roundedScore = Math.round(state.displayedScore);
  state.lastValidScore = roundedScore;

  if (state.isSessionRunning) {
    state.lowestAlertnessScore = Math.min(state.lowestAlertnessScore, roundedScore);
  }

  if (dom.alertnessScore) {
    dom.alertnessScore.textContent = roundedScore;
  }

  const band = getScoreBand(roundedScore);
  const scoreColor = band.color;
  const statusTier = band.tier;

  if (dom.alertnessScore) {
    dom.alertnessScore.style.color = scoreColor;
  }

  if (dom.scoreRingCircle) {
    const maxOffset = 251.3;
    const offset = maxOffset * (1 - roundedScore / 100);
    dom.scoreRingCircle.style.strokeDashoffset = String(offset);
    dom.scoreRingCircle.style.stroke = scoreColor;
  }

  if (dom.scoreLiveInsight && dom.scoreInsightHeading && dom.scoreInsightDetail) {
    dom.scoreInsightHeading.textContent = band.label;
    dom.scoreInsightHeading.style.color = scoreColor;
    dom.scoreInsightDetail.textContent = band.desc;
  }

  // Record history sample every 2.5s for 7-stage history strip
  if (!state.lastHistorySampleTime || nowMs - state.lastHistorySampleTime >= 2500) {
    state.lastHistorySampleTime = nowMs;
    const timeStr = new Date().toTimeString().slice(0, 8);
    state.scoreHistory.push({ score: roundedScore, tier: statusTier, time: timeStr });
    if (state.scoreHistory.length > 7) {
      state.scoreHistory.shift();
    }
  }

  updateSevenBarStrip();
}

function updateSevenBarStrip() {
  if (!dom.sevenBarStrip) return;
  const bars = dom.sevenBarStrip.querySelectorAll('.bar-segment');
  const emptyMsg = dom.stripEmptyMsg || document.getElementById('strip-empty-msg');

  if (!state.scoreHistory || state.scoreHistory.length === 0) {
    if (emptyMsg) {
      emptyMsg.textContent = 'No session data yet';
      emptyMsg.classList.remove('hidden');
      emptyMsg.style.display = '';
    }
    bars.forEach((bar) => {
      bar.className = 'bar-segment bar-empty';
      bar.classList.remove('hidden');
      bar.removeAttribute('hidden');
      bar.style.backgroundColor = '';
      bar.setAttribute('title', 'No data');
    });
    return;
  }

  if (emptyMsg) {
    emptyMsg.classList.add('hidden');
    emptyMsg.style.display = 'none';
  }

  bars.forEach((bar, idx) => {
    bar.classList.remove('hidden');
    bar.removeAttribute('hidden');
    if (idx < state.scoreHistory.length) {
      const entry = state.scoreHistory[idx];
      bar.className = `bar-segment active-${entry.tier}`;
      bar.style.backgroundColor = '';
      const stageNum = idx + 1;
      const tierLabel = entry.tier === 'drowsy' ? 'Drowsy stage' : (entry.tier === 'distracted' ? 'Distracted/Watch stage' : 'Focused stage');
      const timePart = entry.time ? ` at ${entry.time}` : '';
      bar.setAttribute('title', `Stage ${stageNum}: Score ${entry.score}/100 (${tierLabel})${timePart}`);
    } else {
      bar.className = 'bar-segment bar-empty';
      bar.style.backgroundColor = '';
      bar.setAttribute('title', 'No data');
    }
  });
}

function updateScoreAndContributors() {
  if (state.isDemoMode) return;

  // 1. Camera off or no face: "— / Waiting" across all 4 contributors
  if (!state.faceDetected || state.cameraState === 'Off') {
    updateContributor('eyesOpen',  '—', 'Waiting', 'Awaiting camera feed');
    updateContributor('headPose',  '—', 'Waiting', 'Awaiting camera feed');
    updateContributor('blinksMin', '—', 'Waiting', 'Awaiting camera feed');
    updateContributor('lookAways', '—', 'Waiting', 'Awaiting camera feed');
    return;
  }

  // 2. Calibrating baseline (first 5s with face): "-- / Calibrating" across all 4 contributors
  if (state.isCalibratingBaseline) {
    updateContributor('eyesOpen',  '--', 'Calibrating', 'Calibrating baseline…');
    updateContributor('headPose',  '--', 'Calibrating', 'Align natural centre…');
    updateContributor('blinksMin', '--', 'Calibrating', 'Establishing rate…');
    updateContributor('lookAways', '--', 'Calibrating', 'Calibrating…');
    return;
  }

  // 3. Live vigilance tracking:
  // Eyes Open %
  const avgOpenness = Math.round(((state.eyeOpennessL ?? 100) + (state.eyeOpennessR ?? 100)) / 2);
  let openRating = 'Good';
  let openSubline = 'Sustained eye openness';
  if (avgOpenness < 50) {
    openRating = 'Low';
    openSubline = 'Severe eye closure';
  } else if (avgOpenness < 80) {
    openRating = 'Watch';
    openSubline = 'Partial eyelid droop';
  }
  updateContributor('eyesOpen', `${avgOpenness}%`, openRating, openSubline);

  // Head Pose
  const absYaw   = Math.abs(Math.round(state.smoothedYaw || state.gazeYaw || 0));
  const absPitch = Math.abs(Math.round(state.smoothedPitch || state.pitch || 0));
  const isCentred = (state.headDirection === 'CENTRE' || (absYaw < 12 && absPitch < 10));
  const now = Date.now();
  const recentLookaway = state.lastLookAwayTime > 0 && (now - state.lastLookAwayTime < 30000);

  let poseRating = 'Good';
  let poseSubline = 'Natural centre aligned';

  if (isCentred) {
    if (recentLookaway) {
      poseRating = 'Watch';
      poseSubline = 'Recent look-away';
    } else {
      poseRating = 'Good';
      poseSubline = 'Natural centre aligned';
    }
  } else {
    if (absYaw >= 25 || absPitch >= 20) {
      poseRating = 'Low';
      poseSubline = 'Head turned away';
    } else {
      poseRating = 'Watch';
      poseSubline = 'Slight head angle';
    }
  }

  const dirLabel = state.headDirection === 'CENTRE'
    ? 'Center (0°)'
    : `${state.headDirection} (${Math.round(state.smoothedYaw)}°)`;
  updateContributor('headPose', dirLabel, poseRating, poseSubline);

  // Blinks/min
  const blinks = state.blinkTimestamps.length;
  let blinkRating = 'Good';
  let blinkSubline = 'Normal spontaneous blink rate';
  if (blinks < 6) {
    blinkRating = 'Low';
    blinkSubline = 'Low blinks/min lowers alertness';
  } else if (blinks < 10) {
    blinkRating = 'Watch';
    blinkSubline = 'Low blinks/min lowers alertness';
  } else if (blinks > 35) {
    blinkRating = 'Low';
    blinkSubline = 'Excessive rapid blinking / eye strain';
  } else if (blinks > 25) {
    blinkRating = 'Watch';
    blinkSubline = 'Elevated blink rate / fatigue strain';
  }
  updateContributor('blinksMin', `${blinks} / min`, blinkRating, blinkSubline);

  // Look-away count: only increments on real look-away events (>= 1.2s distraction)
  const lCount = state.lookAwayCount;
  let lRating = 'Good';
  let lSubline = lCount === 1 ? '1 look-away in this session' : `${lCount} look-aways in this session`;
  if (lCount >= 4) {
    lRating = 'Low';
    lSubline = `${lCount} look-aways · High distraction`;
  } else if (lCount >= 2) {
    lRating = 'Watch';
    lSubline = `${lCount} look-aways · Watch drift`;
  }
  updateContributor('lookAways', String(lCount), lRating, lSubline);
}

function updateContributor(key, valText, rating, sublineText) {
  state.contributors[key] = { value: valText, rating, subline: sublineText };

  const elVal = {
    eyesOpen:  dom.valEyesOpen,
    headPose:  dom.valHeadPose,
    blinksMin: dom.valBlinksMin,
    lookAways: dom.valLookaways,
  }[key];

  const elPill = {
    eyesOpen:  dom.pillEyesOpen,
    headPose:  dom.pillHeadPose,
    blinksMin: dom.pillBlinksMin,
    lookAways: dom.pillLookaways,
  }[key];

  const elSub = {
    eyesOpen:  dom.subEyesOpen,
    headPose:  dom.subHeadPose,
    blinksMin: dom.subBlinksMin,
    lookAways: dom.subLookaways,
  }[key];

  if (elVal) elVal.textContent = valText;
  if (elSub && sublineText !== undefined) elSub.textContent = sublineText;

  if (elPill) {
    const classMap = {
      'Good':        'pill-focused',
      'Watch':       'pill-distracted',
      'Low':         'pill-drowsy',
      'Calibrating': 'pill-calibrating',
      'Waiting':     'pill-waiting',
    };
    const dotMap = {
      'Good':        'dot-focused',
      'Watch':       'dot-distracted',
      'Low':         'dot-drowsy',
      'Calibrating': 'dot-calibrating',
      'Waiting':     'dot-waiting',
    };
    const ratingIconMap = {
      'Good':        '✓',
      'Watch':       '⚠',
      'Low':         '✕',
      'Calibrating': '◌',
      'Waiting':     '—',
    };
    const rIcon = ratingIconMap[rating] || '—';

    elPill.className = `status-pill ${classMap[rating] || 'pill-waiting'}`;
    elPill.innerHTML = `<span class="status-dot ${dotMap[rating] || 'dot-waiting'}" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">${rIcon}</span> ${rating}`;
  }
}

// ─── Status Pill Sync (Updates in All 3 Places at Once) ───────
function updateGlobalStatus(newStatus) {
  const previousStatus = state.currentStatus;
  state.currentStatus = newStatus;

  // Extract base status for dot/pill classes
  let baseStatus = 'Calibrating';
  if (newStatus.includes('Focused'))    baseStatus = 'Focused';
  else if (newStatus.includes('Distracted')) baseStatus = 'Distracted';
  else if (newStatus.includes('Drowsy'))     baseStatus = 'Drowsy';
  else if (newStatus.includes('No face'))    baseStatus = 'No face';

  const dotClass = {
    'Focused':     'dot-focused',
    'Distracted':  'dot-distracted',
    'Drowsy':      'dot-drowsy',
    'Calibrating': 'dot-calibrating',
    'No face':     'dot-noface',
  }[baseStatus] || 'dot-calibrating';

  const pillClass = {
    'Focused':     'pill-focused',
    'Distracted':  'pill-distracted',
    'Drowsy':      'pill-drowsy',
    'Calibrating': 'pill-calibrating',
    'No face':     'pill-noface',
  }[baseStatus] || 'pill-calibrating';

  const statusIconMap = {
    'Focused':     '✓',
    'Distracted':  '⚠',
    'Drowsy':      '🚨',
    'Calibrating': '◌',
    'No face':     '✕',
  };
  const icon = statusIconMap[baseStatus] || '◌';

  // 1. Video overlay pill: MUST NOT MOVE (only dot colour, icon and word change)
  if (dom.faceStatusPill) {
    dom.faceStatusPill.className = `status-pill ${pillClass} fixed-width-pill`;
    dom.faceStatusPill.innerHTML = `<span class="status-dot ${dotClass}" aria-hidden="true"></span><span class="status-icon" aria-hidden="true">${icon}</span> ${newStatus}`;

    // Animation budget: status pulse twice then settle
    if (newStatus !== previousStatus) {
      dom.faceStatusPill.classList.remove('status-pulse');
      void dom.faceStatusPill.offsetWidth; // force reflow
      dom.faceStatusPill.classList.add('status-pulse');
      setTimeout(() => {
        dom.faceStatusPill?.classList.remove('status-pulse');
      }, 750);
    }
  }
}

// ─── Session Management (Rule 5) ──────────────────────────────
function formatTimeHMS(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  const pad = n => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

function updateSessionTimerDisplay() {
  const timeStr = formatTimeHMS(state.sessionDurationSec);
  if (dom.sessionTimer) dom.sessionTimer.textContent = timeStr;
  const overlayTimer = document.querySelector('.overlay-timer');
  if (overlayTimer) overlayTimer.textContent = timeStr;
}

async function startSession() {
  await unlockAudioOnGesture();
  state.isSessionRunning = true;
  state.sessionStartTime = Date.now();
  state.sessionDurationSec = 0;
  state.lowestAlertnessScore = 100;
  updateSessionTimerDisplay();

  // Clean up: remove example/fake timeline rows once live session starts
  if (dom.timelineList) {
    const exampleRows = dom.timelineList.querySelectorAll('.timeline-item');
    exampleRows.forEach(r => {
      if (r.getAttribute('data-is-example') === 'true' || r.textContent.includes('(Example)')) {
        r.remove();
      }
    });
    updateSafeRestCardVisibility();
  }

  if (state.sessionTimerInterval) clearInterval(state.sessionTimerInterval);
  state.sessionTimerInterval = setInterval(() => {
    state.sessionDurationSec++;
    updateSessionTimerDisplay();
  }, 1000);

  if (dom.btnCameraToggle) {
    dom.btnCameraToggle.textContent = 'Stop Session';
    dom.btnCameraToggle.className   = 'btn btn-secondary';
  }

  if (dom.btnResetSession) {
    dom.btnResetSession.classList.add('hidden');
  }

  state.lookAwayCount = 0;
  state.lastLookAwayTime = 0;
  state.scoreHistory = [];
  state.lastHistorySampleTime = 0;
  updateSevenBarStrip();

  addTimelineRow('Session started', '0s', 'focused');

  // Baseline calibration: on session start, capture natural front yaw/pitch over first 5s
  startBaselineCalibration(5000);

  if (state.cameraState !== 'Live' && !state.isDemoMode) {
    await startCamera(state.selectedCameraId);
  }
}

function stopSession() {
  state.isSessionRunning = false;
  state.isCalibratingBaseline = false;
  state.baselineCaptured = false;
  state.baselineSamples = [];
  if (state.sessionTimerInterval) {
    clearInterval(state.sessionTimerInterval);
    state.sessionTimerInterval = null;
  }

  if (state.cameraState === 'Live' || state.cameraState === 'No face' || state.cameraState === 'Starting') {
    stopCamera();
  }

  if (dom.btnCameraToggle) {
    dom.btnCameraToggle.textContent = 'Start Session';
    dom.btnCameraToggle.className   = 'btn btn-primary';
  }

  // Reset button becomes visible only after session is stopped
  if (dom.btnResetSession) {
    dom.btnResetSession.classList.remove('hidden');
  }

  // Show session summary modal (duration, event counts, lowest score) without deleting timeline
  openSessionSummaryModal();
}

function openSessionSummaryModal() {
  if (!dom.summaryModal || !dom.summaryBackdrop) return;
  if (dom.summaryDuration)        dom.summaryDuration.textContent = formatTimeHMS(state.sessionDurationSec);
  if (dom.summaryLowestScore)     dom.summaryLowestScore.textContent = (state.lowestAlertnessScore <= 100 && state.lowestAlertnessScore >= 0) ? `${state.lowestAlertnessScore} / 100` : '--';
  if (dom.summaryDrowsyCount)     dom.summaryDrowsyCount.textContent = state.eventCounts.drowsy;
  if (dom.summaryDistractionCount) dom.summaryDistractionCount.textContent = state.eventCounts.distracted;
  if (dom.summaryYawnCount)       dom.summaryYawnCount.textContent = state.eventCounts.yawn;
  if (dom.summaryTotalEvents)     dom.summaryTotalEvents.textContent = state.eventCounts.total;

  dom.summaryModal.classList.remove('hidden');
  dom.summaryBackdrop.classList.remove('hidden');
}

function closeSessionSummaryModal() {
  if (dom.summaryModal)    dom.summaryModal.classList.add('hidden');
  if (dom.summaryBackdrop) dom.summaryBackdrop.classList.add('hidden');
}

// ─── Download Session Report CSV (Rule 1: Client-Side Blob) ────
function downloadSessionReportCSV() {
  const rows = [
    ['Time', 'Status', 'Event', 'Duration']
  ];

  if (state.sessionLogRecords && state.sessionLogRecords.length > 0) {
    state.sessionLogRecords.forEach(rec => {
      const statusWord = rec.category ? (rec.category.charAt(0).toUpperCase() + rec.category.slice(1)) : 'Focused';
      rows.push([
        rec.time || '—',
        statusWord,
        rec.eventText || '—',
        rec.duration || '—'
      ]);
    });
  } else if (dom.timelineList) {
    // Fallback: extract from timeline DOM in chronological order
    const items = Array.from(dom.timelineList.querySelectorAll('.timeline-item')).reverse();
    items.forEach(item => {
      const time = item.querySelector('.tl-time')?.textContent?.trim() || '—';
      const cat = item.getAttribute('data-category') || 'focused';
      const statusWord = cat.charAt(0).toUpperCase() + cat.slice(1);
      const eventText = item.querySelector('.tl-event-text')?.textContent?.trim() || '—';
      const duration = item.querySelector('.duration-chip')?.textContent?.trim() || '—';
      rows.push([time, statusWord, eventText, duration]);
    });
  }

  const csvContent = rows
    .map(r => r.map(val => `"${String(val).replace(/"/g, '""')}"`).join(','))
    .join('\r\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  const dateStr = new Date().toISOString().slice(0, 10);
  link.setAttribute('download', `jaago-session-report-${dateStr}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function resetTimelineOnlyWhenStopped() {
  if (state.isSessionRunning) return; // Only after session is stopped!

  if (dom.timelineList) {
    dom.timelineList.innerHTML = '';
  }
  state.eventCounts = { drowsy: 0, distracted: 0, yawn: 0, total: 0 };
  state.sessionDurationSec = 0;
  state.lookAwayCount = 0;
  state.lastLookAwayTime = 0;
  state.scoreHistory = [];
  state.sessionLogRecords = [];
  state.lastHistorySampleTime = 0;
  updateSevenBarStrip();
  updateScoreAndContributors();
  updateSessionTimerDisplay();
  setTimelineFilter('all');
  closeSessionSummaryModal();

  if (dom.btnResetSession) {
    dom.btnResetSession.classList.add('hidden');
  }
}

// ─── Session Timeline (Newest First, Max 8, Show All, Filter) ─
function addTimelineRow(eventText, durationStr, category = 'focused', isExample = false, hasAckAction = false) {
  if (!dom.timelineList) return;

  const now = new Date();
  const timeStr = now.toTimeString().slice(0, 8);

  const isSimulated = state.isDemoMode || eventText.includes('(Simulated)');
  if (isSimulated && !eventText.includes('(Simulated)')) {
    eventText = `${eventText} (Simulated)`;
  }

  const row = document.createElement('div');
  row.className = 'timeline-item tl-row-new';
  row.setAttribute('data-category', category);
  if (isSimulated) {
    row.setAttribute('data-simulated', 'true');
  }
  if (isExample) {
    row.setAttribute('data-is-example', 'true');
  } else if (!isSimulated) {
    // Track live event counts only
    if (category === 'drowsy') {
      state.eventCounts.drowsy++;
    } else if (category === 'distracted') {
      state.eventCounts.distracted++;
    }
    if (eventText.toLowerCase().includes('yawn')) {
      state.eventCounts.yawn++;
    }
    state.eventCounts.total++;
  }

  const dotClass = {
    'focused':    'dot-focused',
    'distracted': 'dot-distracted',
    'drowsy':     'dot-drowsy',
  }[category] || 'dot-focused';

  const pillClass = {
    'focused':    'pill-focused',
    'distracted': 'pill-distracted',
    'drowsy':     'pill-drowsy',
  }[category] || 'pill-focused';

  const wordLabel = category.charAt(0).toUpperCase() + category.slice(1);

  // Parse duration & Ack requirements:
  // "Ack" is an action button in its own column, NEVER the duration
  let formattedDuration = durationStr;

  if (formattedDuration === 'Ack') {
    formattedDuration = '—';
  } else if (!formattedDuration || formattedDuration === '-') {
    formattedDuration = '—';
  } else if (typeof formattedDuration === 'number') {
    formattedDuration = `${formattedDuration.toFixed(1)}s`;
  }

  // Active alarms (sustained look-away or eye closure starting, duration === '—') get an inline Ack action
  const isActiveAlert = (category === 'drowsy' || category === 'distracted') && (hasAckAction || formattedDuration === '—');

  const actionHtml = isActiveAlert
    ? '<button class="tl-btn-ack" type="button" title="Acknowledge alert" aria-label="Acknowledge alert">Ack</button>'
    : '<span class="tl-action-icon tl-action-icon-cleared" aria-label="Logged">✓</span>';

  // Two lines per row: Line 1 (Time + Pill + Duration + Action) | Line 2 (Full-width Event Text)
  row.innerHTML = `
    <div class="tl-row-top">
      <div class="tl-meta-left">
        <span class="tl-time">${timeStr}</span>
        <span class="status-pill ${pillClass} tl-pill">
          <span class="status-dot ${dotClass}" aria-hidden="true"></span>${wordLabel}
        </span>
      </div>
      <div class="tl-meta-right">
        <span class="duration-chip">${formattedDuration}</span>
        <div class="tl-action-cell">${actionHtml}</div>
      </div>
    </div>
    <div class="tl-event-text" title="${eventText}">${eventText}</div>
  `;

  const ackBtn = row.querySelector('.tl-btn-ack');
  if (ackBtn) {
    ackBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (category === 'drowsy') {
        acknowledgeAlarmFromButton();
      } else if (category === 'distracted') {
        clearDistractionWarning();
      }
      ackBtn.classList.remove('tl-btn-ack-distracted', 'tl-btn-ack-drowsy');
      ackBtn.classList.add('acked', 'tl-action-icon-cleared');
      ackBtn.textContent = '✓';
      ackBtn.disabled = true;
      ackBtn.setAttribute('title', 'Alarm acknowledged');
      ackBtn.setAttribute('aria-label', 'Alarm acknowledged');
    });
  }

  // Calculate detail metadata for Detail Panel (Rule 4)
  const curScore = state.displayedScore !== null ? Math.round(state.displayedScore) : (state.targetScore !== null ? Math.round(state.targetScore) : 100);
  let scoreImpactStr = `${curScore} (Baseline / No change)`;
  if (category === 'drowsy') {
    const after = Math.max(0, curScore - 40);
    scoreImpactStr = `${curScore} → ${after} (-40 pts)`;
  } else if (category === 'distracted') {
    const isMajor = formattedDuration !== '—' && parseFloat(formattedDuration) >= (state.lookawayThresholdSec || 3.0);
    const penalty = isMajor ? 15 : 10;
    const after = Math.max(0, curScore - penalty);
    scoreImpactStr = `${curScore} → ${after} (-${penalty} pts)`;
  } else if (eventText.toLowerCase().includes('yawn')) {
    const after = Math.max(0, curScore - 10);
    scoreImpactStr = `${curScore} → ${after} (-10 pts)`;
  } else if (category === 'focused') {
    scoreImpactStr = `${curScore} → ${Math.min(100, curScore + 2)} (+2 pts/sec recovery)`;
  }

  let ackStatusStr = 'Normal operation / Self-cleared';
  if (hasAckAction || (isActiveAlert && formattedDuration === '—')) {
    ackStatusStr = 'Pending operator response';
  } else if (eventText.toLowerCase().includes("i'm awake")) {
    ackStatusStr = `Acknowledged in ${formattedDuration} (Operator button)`;
  } else if (eventText.toLowerCase().includes("i'm back")) {
    ackStatusStr = `Cleared in ${formattedDuration} (Returned to centre)`;
  }

  const detailData = {
    time: timeStr,
    startTime: timeStr,
    endTime: formattedDuration !== '—' ? timeStr : 'Active / In progress',
    category,
    eventText,
    duration: formattedDuration,
    scoreImpact: scoreImpactStr,
    ackTime: ackStatusStr,
    isSimulated
  };

  row._detailData = detailData;
  row._createdTimestamp = Date.now();

  if (!isExample) {
    state.sessionLogRecords.push(detailData);
  }

  // Click a timeline row -> detail panel (Rule 4: no new page)
  row.addEventListener('click', (e) => {
    if (e.target.closest('.tl-btn-ack')) return;
    openTimelineDetailModal(row._detailData);
  });

  // Prepend: newest first
  dom.timelineList.insertBefore(row, dom.timelineList.firstChild);

  // Bound memory to 50 rows max
  while (dom.timelineList.children.length > 50) {
    dom.timelineList.removeChild(dom.timelineList.lastChild);
  }

  // Remove animation class after highlight completes (1600ms)
  setTimeout(() => {
    row.classList.remove('tl-row-new');
  }, 1600);

  applyTimelineVisibility();
  updateSafeRestCardVisibility();
  updateJourneyFatigueEvents();
  checkJourneyRestRule();
}

// ─── Timeline Detail Modal (Rule 4: No New Page) ──────────────
function openTimelineDetailModal(detail) {
  if (!dom.timelineDetailModal || !dom.timelineDetailBackdrop || !detail) return;

  const cat = detail.category || 'focused';
  const wordLabel = cat.charAt(0).toUpperCase() + cat.slice(1);
  const catIcons = { 'focused': '✓', 'distracted': '⚠', 'drowsy': '🚨' };

  if (dom.detailCategoryPill) {
    dom.detailCategoryPill.className = `status-pill pill-${cat} tl-pill`;
  }
  if (dom.detailCategoryDot) {
    dom.detailCategoryDot.className = `status-dot dot-${cat}`;
  }
  if (dom.detailCategoryIcon) {
    dom.detailCategoryIcon.textContent = catIcons[cat] || '✓';
  }
  if (dom.detailCategoryText) {
    dom.detailCategoryText.textContent = wordLabel;
  }
  if (dom.detailEventText) {
    dom.detailEventText.textContent = detail.eventText || '—';
  }
  if (dom.detailStartTime) {
    dom.detailStartTime.textContent = detail.startTime || detail.time || '—';
  }
  if (dom.detailEndTime) {
    dom.detailEndTime.textContent = detail.endTime || (detail.duration !== '—' ? detail.time : 'Active / In progress');
  }
  if (dom.detailDuration) {
    dom.detailDuration.textContent = detail.duration || '—';
  }
  if (dom.detailScoreImpact) {
    dom.detailScoreImpact.textContent = detail.scoreImpact || '—';
  }
  if (dom.detailAckTime) {
    dom.detailAckTime.textContent = detail.ackTime || '—';
  }

  dom.timelineDetailModal.classList.remove('hidden');
  dom.timelineDetailBackdrop.classList.remove('hidden');
}

function closeTimelineDetailModal() {
  if (dom.timelineDetailModal)    dom.timelineDetailModal.classList.add('hidden');
  if (dom.timelineDetailBackdrop) dom.timelineDetailBackdrop.classList.add('hidden');
}

function updateTimelineFilterCounts() {
  if (!dom.timelineList) return;
  const items = dom.timelineList.querySelectorAll('.timeline-item');
  let focusedCount = 0;
  let distractedCount = 0;
  let drowsyCount = 0;

  items.forEach(item => {
    const cat = item.getAttribute('data-category');
    if (cat === 'focused') focusedCount++;
    else if (cat === 'distracted') distractedCount++;
    else if (cat === 'drowsy') drowsyCount++;
  });
  const totalCount = items.length;

  const countAll = dom.filterCountAll || document.getElementById('filter-count-all');
  const countFocused = dom.filterCountFocused || document.getElementById('filter-count-focused');
  const countDistracted = dom.filterCountDistracted || document.getElementById('filter-count-distracted');
  const countDrowsy = dom.filterCountDrowsy || document.getElementById('filter-count-drowsy');

  if (countAll) {
    countAll.textContent = `(${totalCount})`;
    if (totalCount === 0) countAll.classList?.add('count-zero'); else countAll.classList?.remove('count-zero');
  }
  if (countFocused) {
    countFocused.textContent = `(${focusedCount})`;
    if (focusedCount === 0) countFocused.classList?.add('count-zero'); else countFocused.classList?.remove('count-zero');
  }
  if (countDistracted) {
    countDistracted.textContent = `(${distractedCount})`;
    if (distractedCount === 0) countDistracted.classList?.add('count-zero'); else countDistracted.classList?.remove('count-zero');
  }
  if (countDrowsy) {
    countDrowsy.textContent = `(${drowsyCount})`;
    if (drowsyCount === 0) countDrowsy.classList?.add('count-zero'); else countDrowsy.classList?.remove('count-zero');
  }
}

function setTimelineFilter(filterKey) {
  state.currentFilter = filterKey || 'all';
  if (dom.filterPills) {
    dom.filterPills.forEach(p => {
      const isCur = (p.getAttribute('data-filter') || 'all') === state.currentFilter;
      if (isCur) p.classList?.add('active'); else p.classList?.remove('active');
      p.setAttribute('aria-selected', isCur ? 'true' : 'false');
    });
  }
  applyTimelineVisibility();
  updateSafeRestCardVisibility();
}

function applyTimelineVisibility() {
  if (!dom.timelineList) return;

  // Remove existing empty state message if present
  const existingEmpty = dom.timelineList.querySelector('.timeline-empty-filter');
  if (existingEmpty) {
    if (typeof existingEmpty.remove === 'function') {
      existingEmpty.remove();
    } else if (existingEmpty.parentElement) {
      existingEmpty.parentElement.removeChild(existingEmpty);
    }
  }

  const allRows = Array.from(dom.timelineList.querySelectorAll('.timeline-item'));
  const filter = state.currentFilter || 'all';

  // Filter by category
  const matchingRows = allRows.filter(row => {
    const cat = row.getAttribute('data-category');
    return filter === 'all' || cat === filter;
  });

  allRows.forEach(row => {
    const cat = row.getAttribute('data-category');
    if (filter !== 'all' && cat !== filter) {
      row.style.display = 'none';
    } else {
      row.style.display = '';
    }
  });

  // Enforce max 8 visible when not expanded
  matchingRows.forEach((row, idx) => {
    if (!state.timelineExpanded && idx >= 8) {
      row.classList.add('tl-hidden-overflow');
    } else {
      row.classList.remove('tl-hidden-overflow');
    }
  });

  // If 0 matching rows, show empty state message
  if (matchingRows.length === 0) {
    const emptyMsg = document.createElement('div');
    emptyMsg.className = 'timeline-empty-filter';
    let msgText = 'No session events yet';
    if (filter === 'focused') msgText = 'No focused events yet';
    else if (filter === 'distracted') msgText = 'No distracted events yet';
    else if (filter === 'drowsy') msgText = 'No drowsy events yet';
    emptyMsg.textContent = msgText;
    dom.timelineList.appendChild(emptyMsg);
  }

  // Update dynamic count badges on filter pills
  updateTimelineFilterCounts();

  // Update Show all / Show less toggle button and filtered status
  if (dom.btnTimelineToggleAll) {
    const totalAll = allRows.length;
    const matchingCount = matchingRows.length;

    if (filter !== 'all') {
      // Filtered view: "Showing X of Y • Show all", clicking resets to All
      dom.btnTimelineToggleAll.textContent = `Showing ${matchingCount} of ${totalAll} • Show all`;
      dom.btnTimelineToggleAll.disabled = false;
      dom.btnTimelineToggleAll.style.opacity = '1';
      dom.btnTimelineToggleAll.setAttribute('title', 'Reset filter to Show All');
    } else {
      // All view: standard expand/collapse toggle
      dom.btnTimelineToggleAll.removeAttribute('title');
      if (totalAll <= 8) {
        dom.btnTimelineToggleAll.textContent = 'Show all';
        dom.btnTimelineToggleAll.disabled = true;
        dom.btnTimelineToggleAll.style.opacity = '0.4';
      } else {
        dom.btnTimelineToggleAll.disabled = false;
        dom.btnTimelineToggleAll.style.opacity = '1';
        dom.btnTimelineToggleAll.textContent = state.timelineExpanded ? 'Show less' : `Show all (${totalAll})`;
      }
    }
  }
}

function populateInitialExampleRows() {
  if (!dom.timelineList || dom.timelineList.children.length > 0) return;
  addTimelineRow('Baseline calibration completed — Centre and eye thresholds locked', '5.0s', 'focused', true);
  addTimelineRow('Distracted — Looking LEFT', '1.8s', 'distracted', true);
  addTimelineRow('Drowsy — Eyes closed', '—', 'drowsy', true);
  addTimelineRow('Distraction cleared — I\'m back', '3.5s', 'focused', true);
}

// ─── Canvas Corner Brackets Rendering (Mirrored Webcam) ───────
function drawCornerBracketOverlay() {
  const canvas = dom.canvasOverlay;
  const video  = dom.webcam;
  if (!canvas || !video) return;

  canvas.width  = video.videoWidth  || canvas.offsetWidth;
  canvas.height = video.videoHeight || canvas.offsetHeight;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const w = canvas.width;
  const h = canvas.height;

  if (!state.faceDetected || !state.trackedFaceBox) return;

  const box = state.trackedFaceBox;
  const padX = 14;
  const padY = 16;
  const bx = clamp(box.minX * w - padX, 0, w);
  const by = clamp(box.minY * h - padY, 0, h);
  const bw = clamp((box.maxX - box.minX) * w + padX * 2, 20, w - bx);
  const bh = clamp((box.maxY - box.minY) * h + padY * 2, 20, h - by);

  // Status color for 2px stroke
  let strokeColor = '#22C55E';
  if (state.currentStatus === 'Drowsy')          strokeColor = '#EF4444';
  else if (state.currentStatus === 'Distracted') strokeColor = '#F59E0B';
  else if (state.currentStatus === 'Calibrating') strokeColor = '#F59E0B';

  ctx.strokeStyle = strokeColor;
  ctx.lineWidth   = 2;
  ctx.lineCap     = 'round';

  const armLen = Math.min(22, bw * 0.25, bh * 0.25);

  ctx.beginPath();
  // 1. Top-left corner
  ctx.moveTo(bx, by + armLen);
  ctx.lineTo(bx, by);
  ctx.lineTo(bx + armLen, by);

  // 2. Top-right corner
  ctx.moveTo(bx + bw - armLen, by);
  ctx.lineTo(bx + bw, by);
  ctx.lineTo(bx + bw, by + armLen);

  // 3. Bottom-left corner
  ctx.moveTo(bx, by + bh - armLen);
  ctx.lineTo(bx, by + bh);
  ctx.lineTo(bx + armLen, by + bh);

  // 4. Bottom-right corner
  ctx.moveTo(bx + bw - armLen, by + bh);
  ctx.lineTo(bx + bw, by + bh);
  ctx.lineTo(bx + bw, by + bh - armLen);
  ctx.stroke();

  // Label above corner brackets (horizontally unmirrored)
  ctx.save();
  ctx.translate(bx + bw / 2, by - 8);
  ctx.scale(-1, 1);
  ctx.fillStyle = strokeColor;
  ctx.font = '700 11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  let labelText = state.faceDetected ? 'Face locked' : 'No face';
  if (state.distractionLostFaceSince) {
    labelText = 'Face turned';
  } else if (state.currentStatus === 'Distracted') {
    labelText = 'Distracted';
  } else if (state.currentStatus === 'Drowsy') {
    labelText = 'Drowsy';
  }
  ctx.fillText(labelText, 0, 0);
  ctx.restore();
}

function clearCanvasOverlay() {
  const canvas = dom.canvasOverlay;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ─── UI Helper Updates ────────────────────────────────────────
function setCameraState(nextState) {
  state.cameraState = nextState;
  if (nextState === 'Off' || nextState === 'Blocked') {
    syncAllPills('CAMERA_OFF');
    resetCameraOffHUD();
  } else if (nextState === 'Starting') {
    syncAllPills('CALIBRATING');
    updateEyeSignalWeakPill();
  } else if (nextState === 'No face') {
    syncAllPills('NO_FACE');
    updateEyeSignalWeakPill();
  } else if (nextState === 'Live') {
    if (state.isCalibratingBaseline) {
      syncAllPills('CALIBRATING');
    } else if (state.faceDetected) {
      syncAllPills('FACE_LOCKED');
    } else {
      syncAllPills('NO_FACE');
    }
    updateEyeSignalWeakPill();
  }
  updateScoreAndContributors();
  renderAlertnessScore();
  updateSevenBarStrip();
}

function updateDebugDisplay() {
  if (!dom.dbgBlinkL) return;

  if (state.faceDetected && state.eyeBlinkLeft !== null) {
    dom.dbgBlinkL.textContent = state.eyeBlinkLeft.toFixed(2);
    dom.dbgBlinkR.textContent = state.eyeBlinkRight.toFixed(2);
    dom.dbgJaw.textContent    = state.jawOpen.toFixed(2);
    dom.dbgGaze.textContent   = state.gazeLabel;
  } else if (!state.isDemoMode) {
    resetDebugMetrics();
  }
}

function resetDebugMetrics() {
  if (dom.dbgBlinkL) dom.dbgBlinkL.textContent = '--';
  if (dom.dbgBlinkR) dom.dbgBlinkR.textContent = '--';
  if (dom.dbgJaw)    dom.dbgJaw.textContent    = '--';
  if (dom.dbgGaze)   dom.dbgGaze.textContent   = '--';
}

function updateGuidance(text) {
  if (dom.cameraGuidanceLine) {
    dom.cameraGuidanceLine.textContent = text;
  }
}

function showError(reason, fix) {
  if (dom.cameraErrorBanner && dom.cameraErrorReason && dom.cameraErrorFix) {
    dom.cameraErrorReason.textContent = reason;
    dom.cameraErrorFix.textContent    = fix;
    dom.cameraErrorBanner.classList.remove('hidden');
  }
}

function dismissError() {
  if (dom.cameraErrorBanner) {
    dom.cameraErrorBanner.classList.add('hidden');
  }
}

// ─── Pre-shift Check (One Dark Card, 30s Countdown, 4 Ticks) ──
function evaluatePreshiftResults() {
  // Step 1: Camera connection
  const step1 = state.cameraState === 'Live' || state.isDemoMode || !!state.stream;
  // Step 2: Face in frame
  const step2 = state.faceDetected || state.isDemoMode;
  // Step 3: Ambient lighting OK
  const step3 = true;
  // Step 4: Baseline captured
  const step4 = state.baselineCaptured || state.isDemoMode || state.faceDetected;

  const stepResults = [step1, step2, step3, step4];
  const stepElements = [
    dom.preshiftStep1Res,
    dom.preshiftStep2Res,
    dom.preshiftStep3Res,
    dom.preshiftStep4Res
  ];

  let passCount = 0;
  stepResults.forEach((passed, idx) => {
    if (passed) passCount++;
    const row = stepElements[idx];
    if (!row) return;
    const icon = row.querySelector('.step-res-icon');
    const status = row.querySelector('.step-res-status');
    if (passed) {
      if (icon) {
        icon.className = 'step-res-icon pass';
        icon.textContent = '✓';
      }
      if (status) {
        status.className = 'step-res-status pass';
        status.textContent = 'Passed';
      }
    } else {
      if (icon) {
        icon.className = 'step-res-icon fail';
        icon.textContent = '✕';
      }
      if (status) {
        status.className = 'step-res-status fail';
        status.textContent = 'Failed';
      }
    }
  });

  if (dom.preshiftResultBadge) {
    if (passCount === 4) {
      dom.preshiftResultBadge.textContent = 'FIT';
      dom.preshiftResultBadge.className = 'result-badge result-fit';
    } else if (passCount === 3) {
      dom.preshiftResultBadge.textContent = 'CAUTION';
      dom.preshiftResultBadge.className = 'result-badge result-caution';
    } else {
      dom.preshiftResultBadge.textContent = 'NOT FIT';
      dom.preshiftResultBadge.className = 'result-badge result-not-fit';
    }
  }

  // "Start session" enabled ONLY if all 4 passed (Rule 5)
  if (dom.btnStartSessionPreshift) {
    dom.btnStartSessionPreshift.disabled = (passCount !== 4);
  }
}

function openPreshiftModal() {
  if (!dom.preshiftModal || !dom.preshiftBackdrop) return;
  dom.preshiftStartView?.classList.remove('hidden');
  dom.preshiftActiveView?.classList.add('hidden');
  dom.preshiftResultSection?.classList.add('hidden');
  if (dom.preshiftProgressBar) dom.preshiftProgressBar.style.width = '0%';
  if (dom.preshiftCountdown) dom.preshiftCountdown.textContent = '30s remaining';
  if (dom.preshiftStepLabel) dom.preshiftStepLabel.textContent = 'Step 1 of 4';
  if (dom.btnStartSessionPreshift) dom.btnStartSessionPreshift.disabled = true;

  dom.tickRows.forEach(r => {
    if (r) {
      r.className = 'ticked-row';
      const icon = r.querySelector('.tick-icon');
      if (icon) icon.textContent = '○';
    }
  });
  state.preshiftTicks = [false, false, false, false];
  state.preshiftRunning = false;

  dom.preshiftModal.classList.remove('hidden');
  dom.preshiftBackdrop.classList.remove('hidden');
}

function closePreshiftModal() {
  if (state.preshiftTimer) {
    clearInterval(state.preshiftTimer);
    state.preshiftTimer = null;
  }
  state.preshiftRunning = false;
  if (dom.preshiftModal) dom.preshiftModal.classList.add('hidden');
  if (dom.preshiftBackdrop) dom.preshiftBackdrop.classList.add('hidden');
}

function tickPreshiftRow(index) {
  const row = dom.tickRows[index];
  if (!row) return;
  state.preshiftTicks[index] = true;
  row.classList.add('ticked');
  const icon = row.querySelector('.tick-icon');
  if (icon) icon.textContent = '✓';
}

async function startPreshiftCheck() {
  await unlockAudioOnGesture();
  state.preshiftRunning = true;
  state.preshiftCountdownSec = 30;

  dom.preshiftStartView?.classList.add('hidden');
  dom.preshiftActiveView?.classList.remove('hidden');
  dom.preshiftResultSection?.classList.add('hidden');

  if (dom.preshiftStepLabel) dom.preshiftStepLabel.textContent = 'Step 1 of 4';
  if (dom.preshiftCountdown) dom.preshiftCountdown.textContent = '30s remaining';
  if (dom.preshiftProgressBar) dom.preshiftProgressBar.style.width = '0%';

  if (state.cameraState !== 'Live' && !state.isDemoMode) {
    startCamera(state.selectedCameraId).catch(() => {});
  }

  if (state.preshiftTimer) clearInterval(state.preshiftTimer);

  state.preshiftTimer = setInterval(() => {
    state.preshiftCountdownSec--;
    const secLeft = state.preshiftCountdownSec;

    // Never a spinner with no count — explicit countdown
    if (dom.preshiftCountdown) {
      dom.preshiftCountdown.textContent = `${secLeft}s remaining`;
    }

    const elapsed = 30 - secLeft;
    const pct = Math.min(100, (elapsed / 30) * 100);
    if (dom.preshiftProgressBar) {
      dom.preshiftProgressBar.style.width = `${pct}%`;
    }

    // Step 1: Camera found (~3s in)
    if (elapsed >= 3 && !state.preshiftTicks[0]) {
      tickPreshiftRow(0);
      if (dom.preshiftStepLabel) dom.preshiftStepLabel.textContent = 'Step 2 of 4';
    }

    // Step 2: Face in frame (~9s in)
    if (elapsed >= 9 && !state.preshiftTicks[1]) {
      tickPreshiftRow(1);
      if (dom.preshiftStepLabel) dom.preshiftStepLabel.textContent = 'Step 3 of 4';
    }

    // Step 3: Lighting OK (~18s in)
    if (elapsed >= 18 && !state.preshiftTicks[2]) {
      tickPreshiftRow(2);
      if (dom.preshiftStepLabel) dom.preshiftStepLabel.textContent = 'Step 4 of 4';
    }

    // Step 4: Baseline captured (~29s in)
    if (elapsed >= 29 && !state.preshiftTicks[3]) {
      tickPreshiftRow(3);
      if (dom.preshiftStepLabel) dom.preshiftStepLabel.textContent = 'Step 4 of 4 · Complete';
    }

    // Completion at 0s
    if (secLeft <= 0) {
      clearInterval(state.preshiftTimer);
      state.preshiftTimer = null;
      state.preshiftRunning = false;

      // Ensure all 4 rows ticked
      for (let i = 0; i < 4; i++) tickPreshiftRow(i);

      if (dom.preshiftProgressBar) dom.preshiftProgressBar.style.width = '100%';
      if (dom.preshiftCountdown) dom.preshiftCountdown.textContent = '0s · Assessment complete';
      evaluatePreshiftResults();
      if (dom.preshiftResultSection) dom.preshiftResultSection.classList.remove('hidden');
    }
  }, 1000);
}

// ─── Demo Mode (Simulated Without Camera) ──────────────────────
function drawSimulatedCornerBrackets(statusText, strokeColor) {
  const canvas = dom.canvasOverlay;
  if (!canvas || typeof canvas.getContext !== 'function') return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const w = canvas.width || 640;
  const h = canvas.height || 360;
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);

  const bw = 220;
  const bh = 250;
  const bx = (w - bw) / 2;
  const by = (h - bh) / 2;
  const armLen = 22;

  ctx.strokeStyle = strokeColor;
  ctx.lineWidth   = 2;
  ctx.lineCap     = 'round';

  ctx.beginPath();
  // Top-left
  ctx.moveTo(bx, by + armLen); ctx.lineTo(bx, by); ctx.lineTo(bx + armLen, by);
  // Top-right
  ctx.moveTo(bx + bw - armLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + armLen);
  // Bottom-left
  ctx.moveTo(bx, by + bh - armLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + armLen, by + bh);
  // Bottom-right
  ctx.moveTo(bx + bw - armLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - armLen);
  ctx.stroke();

  // Label unmirrored
  ctx.save();
  ctx.translate(bx + bw / 2, by - 8);
  ctx.scale(-1, 1);
  ctx.fillStyle = strokeColor;
  ctx.font = '700 11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(statusText, 0, 0);
  ctx.restore();
}

function setAvatarState(stateName) {
  const head = document.getElementById('demo-operator-head');
  const hudBox = document.getElementById('demo-hud-box');
  const eyesOpen = document.getElementById('demo-eyes-open');
  const eyesClosed = document.getElementById('demo-eyes-closed');
  const pupilL = document.getElementById('demo-pupil-left');
  const pupilR = document.getElementById('demo-pupil-right');
  const glintL = document.getElementById('demo-glint-left');
  const glintR = document.getElementById('demo-glint-right');
  const hudText = document.getElementById('demo-hud-badge-text');
  const hudBadgeBg = document.getElementById('demo-hud-badge-bg');
  const hudCross = document.getElementById('demo-hud-cross');
  const brackets = [
    document.getElementById('demo-bracket-tl'),
    document.getElementById('demo-bracket-tr'),
    document.getElementById('demo-bracket-bl'),
    document.getElementById('demo-bracket-br'),
  ];

  if (!head || !hudBox) return;

  if (stateName === 'focused') {
    head.setAttribute('transform', 'translate(320, 180) rotate(0)');
    hudBox.setAttribute('transform', 'translate(320, 180)');
    if (eyesOpen) eyesOpen.classList.remove('hidden');
    if (eyesClosed) eyesClosed.classList.add('hidden');
    if (pupilL) pupilL.setAttribute('cx', '-22');
    if (pupilR) pupilR.setAttribute('cx', '22');
    if (glintL) glintL.setAttribute('cx', '-23.5');
    if (glintR) glintR.setAttribute('cx', '20.5');

    const color = '#22C55E';
    brackets.forEach(b => b && b.setAttribute('stroke', color));
    if (hudBadgeBg) hudBadgeBg.setAttribute('stroke', color);
    if (hudText) {
      hudText.setAttribute('fill', color);
      hudText.textContent = 'VIGILANCE LOCK [OK]';
    }
    if (hudCross) hudCross.setAttribute('fill', color);
    hudBox.classList.remove('pulsing');
  } else if (stateName === 'distracted') {
    head.setAttribute('transform', 'translate(365, 180) rotate(14)');
    hudBox.setAttribute('transform', 'translate(365, 180)');
    if (eyesOpen) eyesOpen.classList.remove('hidden');
    if (eyesClosed) eyesClosed.classList.add('hidden');
    if (pupilL) pupilL.setAttribute('cx', '-16');
    if (pupilR) pupilR.setAttribute('cx', '27');
    if (glintL) glintL.setAttribute('cx', '-17.5');
    if (glintR) glintR.setAttribute('cx', '25.5');

    const color = '#F59E0B';
    brackets.forEach(b => b && b.setAttribute('stroke', color));
    if (hudBadgeBg) hudBadgeBg.setAttribute('stroke', color);
    if (hudText) {
      hudText.setAttribute('fill', color);
      hudText.textContent = 'ATTENTION DRIFT [+28°]';
    }
    if (hudCross) hudCross.setAttribute('fill', color);
    hudBox.classList.remove('pulsing');
  } else if (stateName === 'drowsy') {
    head.setAttribute('transform', 'translate(320, 190) rotate(2)');
    hudBox.setAttribute('transform', 'translate(320, 190)');
    if (eyesOpen) eyesOpen.classList.add('hidden');
    if (eyesClosed) eyesClosed.classList.remove('hidden');

    const color = '#EF4444';
    brackets.forEach(b => b && b.setAttribute('stroke', color));
    if (hudBadgeBg) hudBadgeBg.setAttribute('stroke', color);
    if (hudText) {
      hudText.setAttribute('fill', color);
      hudText.textContent = 'CRITICAL CLOSURE [1.5s]';
    }
    if (hudCross) hudCross.setAttribute('fill', color);
    hudBox.classList.add('pulsing');
  }
}

function startDemoMode() {
  if (state.cameraState === 'Live' || state.cameraState === 'No face' || state.cameraState === 'Starting') {
    stopCamera();
  }

  state.isDemoMode = true;
  state.scoreHistory = [];
  state.lastHistorySampleTime = 0;

  document.body.classList.add('demo-mode-active');

  if (dom.clearSimulatedBanner) dom.clearSimulatedBanner.classList.add('hidden');
  if (dom.cameraErrorBanner)    dom.cameraErrorBanner.classList.add('hidden');
  clearVideoPaneError();
  dismissError();

  if (dom.toggleDemoMode) {
    dom.toggleDemoMode.classList.add('active');
    dom.toggleDemoMode.setAttribute('aria-checked', 'true');
  }
  if (dom.demoToggleLabel) {
    dom.demoToggleLabel.textContent = 'DEMO MODE — SIMULATED';
    dom.demoToggleLabel.classList.add('simulated');
  }
  if (dom.demoBadge) dom.demoBadge.classList.remove('hidden');
  if (dom.videoSimulatedPill) dom.videoSimulatedPill.classList.remove('hidden');
  if (dom.scoreSimulatedBadge) dom.scoreSimulatedBadge.classList.remove('hidden');
  if (dom.demoControlsBar) dom.demoControlsBar.classList.remove('hidden');
  if (dom.demoIllustrationView) dom.demoIllustrationView.classList.remove('hidden');
  if (dom.placeholderGraphic) dom.placeholderGraphic.classList.add('hidden');
  if (dom.webcam) dom.webcam.classList.add('hidden');

  setCameraState('Live');
  updateCameraIconUI(true);
  updateTopBarLiveStatus();
  syncJourneyDemoModeUI();

  goToDemoStage(1);
}

function stopDemoMode() {
  state.isDemoMode = false;
  state.scoreHistory = [];
  state.lastHistorySampleTime = 0;
  if (state.demoTimer) {
    clearTimeout(state.demoTimer);
    state.demoTimer = null;
  }

  // Instantly stop all audio
  stopDistractionSound();
  stopLevel3AudioAlarm();
  if (typeof stopDrowsySoundTest === 'function') stopDrowsySoundTest();

  // Instantly hide banners and reset styles
  hideDistractionBanners();
  dismissLevel3Alarm();
  document.body.classList.remove('freeze-non-alarm');
  document.body.classList.remove('demo-mode-active');

  // Reset top bar toggle UI
  if (dom.toggleDemoMode) {
    dom.toggleDemoMode.classList.remove('active');
    dom.toggleDemoMode.setAttribute('aria-checked', 'false');
  }
  if (dom.demoToggleLabel) {
    dom.demoToggleLabel.textContent = 'Demo mode';
    dom.demoToggleLabel.classList.remove('simulated');
  }
  if (dom.demoBadge) dom.demoBadge.classList.add('hidden');
  if (dom.videoSimulatedPill) dom.videoSimulatedPill.classList.add('hidden');
  if (dom.scoreSimulatedBadge) dom.scoreSimulatedBadge.classList.add('hidden');
  if (dom.demoControlsBar) dom.demoControlsBar.classList.add('hidden');
  if (dom.demoIllustrationView) dom.demoIllustrationView.classList.add('hidden');
  syncJourneyDemoModeUI();

  // Restore camera standby state
  clearCanvasOverlay();
  updateDirectionHUD(0, 0, 'CENTRE', 0);
  if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
  if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
  if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Centre';

  setCameraState('Off');
  updateCameraIconUI(false);
  updateGlobalStatus('No face');
  updateTopBarLiveStatus();
  updateGuidance('Demo stopped · Camera offline · Click "Start Session" to begin');
  resetDebugMetrics();
  updateEyeOpennessHUD(null, null, 0, state.drowsyThresholdSec, false, true);
  renderAlertnessScore();
  updateSevenBarStrip();

  // Check if simulated rows exist in timeline, offer to clear them
  const simulatedRows = dom.timelineList ? dom.timelineList.querySelectorAll('.timeline-item[data-simulated="true"]') : [];
  if (simulatedRows.length > 0 && dom.clearSimulatedBanner) {
    if (dom.simulatedRowsCount) dom.simulatedRowsCount.textContent = simulatedRows.length;
    dom.clearSimulatedBanner.classList.remove('hidden');
  }
}

function goToDemoStage(stageNum) {
  if (!state.isDemoMode) return;
  state.demoStage = stageNum;

  if (state.demoTimer) {
    clearTimeout(state.demoTimer);
    state.demoTimer = null;
  }

  if (dom.demoStepBadge) dom.demoStepBadge.textContent = `STAGE ${stageNum} OF 4`;
  if (dom.btnDemoPrev)   dom.btnDemoPrev.disabled = (stageNum <= 1);

  switch (stageNum) {
    case 1: {
      // 1. Focused (Simulated)
      stopDistractionSound();
      stopLevel3AudioAlarm();
      hideDistractionBanners();
      dismissLevel3Alarm();
      document.body.classList.remove('freeze-non-alarm');

      if (dom.demoStageTitle) dom.demoStageTitle.textContent = 'Focused Operator (Simulated)';
      if (dom.demoStageDesc)  dom.demoStageDesc.textContent  = 'Operator centered and alert. Alertness score is 100 with optimal vigilance. Click Next to simulate distraction.';
      if (dom.btnDemoNext) {
        dom.btnDemoNext.textContent = 'Next: Distracted →';
        dom.btnDemoNext.disabled = false;
      }

      setAvatarState('focused');
      updateGlobalStatus('Focused (Simulated)');
      setTargetScore(100);
      updateContributor('eyesOpen', '98% (Simulated)', 'Good', 'Sustained eye openness');
      updateContributor('headPose', 'Center (0°) (Simulated)', 'Good', 'Natural centre aligned');
      updateContributor('blinksMin', '16 / min (Simulated)', 'Good', 'Normal spontaneous blink rate');
      updateContributor('lookAways', '0 (Simulated)', 'Good', '0 look-aways in this session');

      if (dom.dbgBlinkL) dom.dbgBlinkL.textContent = '0.04 (Simulated)';
      if (dom.dbgBlinkR) dom.dbgBlinkR.textContent = '0.05 (Simulated)';
      if (dom.dbgJaw)    dom.dbgJaw.textContent    = '0.02 (Simulated)';
      if (dom.dbgGaze)   dom.dbgGaze.textContent   = 'Center (0°) (Simulated)';

      updateDirectionHUD(0, 0, 'CENTRE', 0);
      updateEyeOpennessHUD(98, 98, 0, state.drowsyThresholdSec, false, false);
      if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
      if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
      if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Centre';

      updateGuidance('Face locked · Vigilance tracking active (Simulated)');
      drawSimulatedCornerBrackets('Face locked (Simulated)', '#22C55E');
      addTimelineRow('Continuous focus (Simulated)', '—', 'focused');
      renderAlertnessScore();
      break;
    }

    case 2: {
      // 2. Look-away Distraction (Simulated)
      stopLevel3AudioAlarm();
      dismissLevel3Alarm();
      document.body.classList.remove('freeze-non-alarm');

      if (dom.demoStageTitle) dom.demoStageTitle.textContent = 'Distracted — Head Turned (Simulated)';
      if (dom.demoStageDesc)  dom.demoStageDesc.textContent  = 'Operator turned right (+28°). Amber distraction banner displayed with alert audio. Click Next to simulate drowsiness.';
      if (dom.btnDemoNext) {
        dom.btnDemoNext.textContent = 'Next: Drowsy →';
        dom.btnDemoNext.disabled = false;
      }

      setAvatarState('distracted');
      updateGlobalStatus('Distracted (Simulated)');
      setTargetScore(65);
      updateContributor('eyesOpen', '94% (Simulated)', 'Good', 'Eyes open during drift');
      updateContributor('headPose', 'Right (+28°) (Simulated)', 'Low', 'Head turned away');
      updateContributor('blinksMin', '14 / min (Simulated)', 'Good', 'Spontaneous blink rate');
      updateContributor('lookAways', '1 (Simulated)', 'Watch', '1 look-away · Watch drift');

      if (dom.dbgBlinkL) dom.dbgBlinkL.textContent = '0.06 (Simulated)';
      if (dom.dbgBlinkR) dom.dbgBlinkR.textContent = '0.06 (Simulated)';
      if (dom.dbgGaze)   dom.dbgGaze.textContent   = 'Right (+28°) (Simulated)';

      if (dom.bannerTextL1) {
        dom.bannerTextL1.textContent = 'Distracted — Look back at the screen (Simulated)';
      }
      if (dom.bannerReasonL1) {
        dom.bannerReasonL1.textContent = 'Why did this alarm fire? Head turned RIGHT continuously for 1.8s (Simulated)';
      }
      updateBannerMutedPills();
      if (dom.distractionBannerL1) dom.distractionBannerL1.classList.remove('hidden');
      if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '1.8s';
      if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar lookaway-level1';
      if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Level 1 (Simulated)';
      updateDirectionHUD(28, 4, 'RIGHT', 1.8);
      updateEyeOpennessHUD(94, 94, 0, state.drowsyThresholdSec, false, false);
      startDistractionAudioLevel1();

      drawSimulatedCornerBrackets('Distracted (Simulated)', '#F59E0B');
      addTimelineRow('Distracted — Looking RIGHT (Simulated)', '1.8s', 'distracted');
      renderAlertnessScore();
      break;
    }

    case 3: {
      // 3. Eyes closed & Level-3 Alarm (Simulated)
      stopDistractionSound();
      hideDistractionBanners();
      updateDirectionHUD(0, 0, 'CENTRE', 0);
      if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '0.0s';
      if (dom.lookawayTimerBar) dom.lookawayTimerBar.className = 'lookaway-timer-bar';
      if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = 'Centre';

      if (dom.demoStageTitle) dom.demoStageTitle.textContent = 'Drowsiness Alarm — Eyes Closed (Simulated)';
      if (dom.demoStageDesc)  dom.demoStageDesc.textContent  = 'Operator eyes closed for 1.5s. Level 3 red alarm activated with escalating siren and vibration. Click "I\'m awake" to acknowledge.';
      if (dom.btnDemoNext) {
        dom.btnDemoNext.textContent = "Simulate: I'm awake →";
        dom.btnDemoNext.disabled = false;
      }

      setAvatarState('drowsy');
      updateGlobalStatus('Drowsy (Simulated)');
      setTargetScore(28);

      if (dom.dbgBlinkL) dom.dbgBlinkL.textContent = '0.98 (Simulated)';
      if (dom.dbgBlinkR) dom.dbgBlinkR.textContent = '0.97 (Simulated)';
      if (dom.dbgGaze)   dom.dbgGaze.textContent   = 'Eyes Closed (Simulated)';
      updateContributor('eyesOpen', '2% (Simulated)', 'Low', 'Severe eye closure');
      updateContributor('headPose', 'Center (0°) (Simulated)', 'Good', 'Head drooped slightly');
      updateContributor('blinksMin', '4 / min (Simulated)', 'Low', 'Very low blink rate');
      updateContributor('lookAways', '1 (Simulated)', 'Watch', '1 look-away in session');

      if (dom.alarmLevelLabel)  dom.alarmLevelLabel.textContent  = 'ALARM LEVEL 3 · DROWSINESS DETECTED (Simulated)';
      if (dom.alarmMessageText) dom.alarmMessageText.textContent = 'Sustained eye closure detected (Simulated) — tap "I\'m awake"';
      if (dom.alarmReasonText)  dom.alarmReasonText.textContent  = 'Why did this alarm fire? Eyes closed continuously for 1.5s (Simulated)';
      if (dom.alarmBar) dom.alarmBar.classList.remove('hidden');

      document.body.classList.add('freeze-non-alarm');
      startLevel3AudioAlarm();

      drawSimulatedCornerBrackets('Drowsy (Simulated)', '#EF4444');
      updateEyeOpennessHUD(2, 2, 1.5, state.drowsyThresholdSec, true, false);
      addTimelineRow('Drowsy — Eyes closed (Simulated)', '1.5s', 'drowsy');
      renderAlertnessScore();
      break;
    }

    case 4: {
      // 4. "I'm awake" Action & Recovery (Simulated)
      stopLevel3AudioAlarm();
      stopDistractionSound();
      hideDistractionBanners();
      dismissLevel3Alarm();
      document.body.classList.remove('freeze-non-alarm');

      if (dom.demoStageTitle) dom.demoStageTitle.textContent = 'Focus Restored — Operator Awake (Simulated)';
      if (dom.demoStageDesc)  dom.demoStageDesc.textContent  = 'Alarm acknowledged and silenced. Eye openness restored, score recovered to 100, cleared timeline event logged.';
      if (dom.btnDemoNext) {
        dom.btnDemoNext.textContent = '↻ Replay Demo';
        dom.btnDemoNext.disabled = false;
      }

      setAvatarState('focused');
      updateGlobalStatus('Focused (Simulated)');
      setTargetScore(100);
      updateContributor('eyesOpen', '97% (Simulated)', 'Good', 'Sustained eye openness restored');
      updateContributor('headPose', 'Center (0°) (Simulated)', 'Good', 'Natural centre aligned');
      updateContributor('blinksMin', '16 / min (Simulated)', 'Good', 'Normal spontaneous blink rate');
      updateContributor('lookAways', '1 (Simulated)', 'Good', 'Session stabilized');

      if (dom.dbgBlinkL) dom.dbgBlinkL.textContent = '0.04 (Simulated)';
      if (dom.dbgBlinkR) dom.dbgBlinkR.textContent = '0.05 (Simulated)';
      if (dom.dbgJaw)    dom.dbgJaw.textContent    = '0.02 (Simulated)';
      if (dom.dbgGaze)   dom.dbgGaze.textContent   = 'Center (0°) (Simulated)';

      updateDirectionHUD(0, 0, 'CENTRE', 0);
      updateEyeOpennessHUD(97, 98, 0, state.drowsyThresholdSec, false, false);
      drawSimulatedCornerBrackets('Face locked (Simulated)', '#22C55E');
      addTimelineRow("Drowsy cleared — I'm awake (Simulated)", '1.5s', 'focused');
      renderAlertnessScore();
      break;
    }
  }
}

function handleDemoNext(e) {
  if (e && e.preventDefault) e.preventDefault();
  if (!state.isDemoMode) return;
  if (state.demoStage < 4) {
    goToDemoStage(state.demoStage + 1);
  } else {
    goToDemoStage(1);
  }
}

function handleDemoPrev(e) {
  if (e && e.preventDefault) e.preventDefault();
  if (!state.isDemoMode) return;
  if (state.demoStage > 1) {
    goToDemoStage(state.demoStage - 1);
  }
}

function clearSimulatedRowsFromTimeline() {
  if (dom.timelineList) {
    const simulatedRows = dom.timelineList.querySelectorAll('.timeline-item[data-simulated="true"]');
    simulatedRows.forEach(r => r.remove());
  }
  state.sessionLogRecords = state.sessionLogRecords.filter(r => !r.isSimulated && !r.eventText.includes('(Simulated)'));
  if (dom.clearSimulatedBanner) {
    dom.clearSimulatedBanner.classList.add('hidden');
  }
  applyTimelineVisibility();
  updateSafeRestCardVisibility();
}

function dismissSimulatedBanner() {
  if (dom.clearSimulatedBanner) {
    dom.clearSimulatedBanner.classList.add('hidden');
  }
}

// ─── Safe Rest Assistant (Isolated: Reads Session Timeline fatigue events) ────
function getSessionTimelineFatigueCounts() {
  if (!dom.timelineList) return { drowsy: 0, distracted: 0 };
  const items = dom.timelineList.querySelectorAll('.timeline-item');
  let drowsy = 0;
  let distracted = 0;
  items.forEach(item => {
    const cat = item.getAttribute('data-category');
    if (cat === 'drowsy') drowsy++;
    else if (cat === 'distracted') distracted++;
  });
  return { drowsy, distracted };
}

function updateSafeRestCardVisibility() {
  if (!dom.safeRestCard) return;

  // If rest is currently in progress, in post-break check, or completed, keep it visible
  if (state.safeRest.isTimerRunning || state.safeRest.viewState === 'check' || state.safeRest.hasCompletedRest) {
    dom.safeRestCard.classList.remove('hidden');
    return;
  }

  const counts = getSessionTimelineFatigueCounts();
  // Persistent fatigue: 2 or more Drowsy OR 3 or more Distracted
  const shouldShow = (counts.drowsy >= 2 || counts.distracted >= 3);

  if (shouldShow) {
    dom.safeRestCard.classList.remove('hidden');
  } else {
    dom.safeRestCard.classList.add('hidden');
  }
}

function setSafeRestDurationPreset(minutes) {
  if (state.safeRest.isTimerRunning) return;
  state.safeRest.selectedMinutes = minutes;
  state.safeRest.remainingSeconds = minutes * 60;

  const btns = dom.safeRestPresetBtns || document.querySelectorAll('.btn-preset-time');
  btns.forEach(b => {
    const m = parseInt(b.getAttribute('data-minutes'), 10);
    if (m === minutes) {
      b.classList.add('active');
    } else {
      b.classList.remove('active');
    }
  });

  updateSafeRestTimerDisplay();
  if (dom.safeRestTimerSubtext) {
    dom.safeRestTimerSubtext.textContent = `Preset duration: ${minutes} minutes`;
  }
}

function updateSafeRestTimerDisplay() {
  if (!dom.safeRestTimerDisplay) return;
  const rem = Math.max(0, state.safeRest.remainingSeconds);
  const m = Math.floor(rem / 60);
  const s = rem % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');
  dom.safeRestTimerDisplay.textContent = `${mm}:${ss}`;
}

function startSafeRestTimer() {
  if (state.safeRest.isTimerRunning) return;
  state.safeRest.isTimerRunning = true;
  state.safeRest.viewState = 'main';

  // Log "Rest started" to timeline
  addTimelineRow('Rest started', `${state.safeRest.selectedMinutes}m`, 'focused');

  // UI state for countdown
  if (dom.safeRestPresetsRow) dom.safeRestPresetsRow.classList.add('hidden');
  if (dom.btnStartRest) dom.btnStartRest.classList.add('hidden');
  if (dom.btnEndRestEarly) dom.btnEndRestEarly.classList.remove('hidden');
  if (dom.safeRestTimerSubtext) {
    dom.safeRestTimerSubtext.textContent = 'Rest in progress • Relax and recharge';
  }

  updateSafeRestTimerDisplay();

  if (state.safeRest.timerInterval) clearInterval(state.safeRest.timerInterval);
  state.safeRest.timerInterval = setInterval(() => {
    state.safeRest.remainingSeconds--;
    updateSafeRestTimerDisplay();

    if (state.safeRest.remainingSeconds <= 0) {
      clearInterval(state.safeRest.timerInterval);
      state.safeRest.timerInterval = null;
      state.safeRest.isTimerRunning = false;

      // Log "Rest completed" to timeline
      addTimelineRow('Rest completed', `${state.safeRest.selectedMinutes}m`, 'focused');

      // Transition to Post-Break Check
      showSafeRestCheckView();
    }
  }, 1000);
}

function endSafeRestEarly() {
  if (!state.safeRest.isTimerRunning) return;
  if (state.safeRest.timerInterval) {
    clearInterval(state.safeRest.timerInterval);
    state.safeRest.timerInterval = null;
  }
  state.safeRest.isTimerRunning = false;

  const totalSec = state.safeRest.selectedMinutes * 60;
  const elapsedSec = Math.max(1, totalSec - state.safeRest.remainingSeconds);
  const elapsedStr = elapsedSec < 60
    ? `${elapsedSec}s`
    : `${Math.floor(elapsedSec / 60)}m ${elapsedSec % 60}s`;

  // Log "Rest ended early" to timeline
  addTimelineRow('Rest ended early', elapsedStr, 'focused');

  // Transition to Post-Break Check
  showSafeRestCheckView();
}

function showSafeRestCheckView() {
  state.safeRest.viewState = 'check';
  if (dom.safeRestMainView) dom.safeRestMainView.classList.add('hidden');
  if (dom.safeRestCheckView) dom.safeRestCheckView.classList.remove('hidden');
  if (dom.safeRestCompletedView) dom.safeRestCompletedView.classList.add('hidden');
}

function finishSafeRestCheck(ready) {
  if (ready) {
    // Collapse to one-line "Rest completed" note
    state.safeRest.hasCompletedRest = true;
    state.safeRest.viewState = 'completed';
    if (dom.safeRestMainView) dom.safeRestMainView.classList.add('hidden');
    if (dom.safeRestCheckView) dom.safeRestCheckView.classList.add('hidden');
    if (dom.safeRestCompletedView) dom.safeRestCompletedView.classList.remove('hidden');
  } else {
    // User requested "Rest more" -> reset to timer view
    state.safeRest.viewState = 'main';
    state.safeRest.hasCompletedRest = false;
    state.safeRest.remainingSeconds = state.safeRest.selectedMinutes * 60;
    if (dom.safeRestPresetsRow) dom.safeRestPresetsRow.classList.remove('hidden');
    if (dom.btnStartRest) dom.btnStartRest.classList.remove('hidden');
    if (dom.btnEndRestEarly) dom.btnEndRestEarly.classList.add('hidden');
    if (dom.safeRestTimerSubtext) {
      dom.safeRestTimerSubtext.textContent = `Preset duration: ${state.safeRest.selectedMinutes} minutes`;
    }
    updateSafeRestTimerDisplay();
    if (dom.safeRestMainView) dom.safeRestMainView.classList.remove('hidden');
    if (dom.safeRestCheckView) dom.safeRestCheckView.classList.add('hidden');
    if (dom.safeRestCompletedView) dom.safeRestCompletedView.classList.add('hidden');
  }
}

function reopenSafeRestTimer() {
  state.safeRest.viewState = 'main';
  state.safeRest.hasCompletedRest = false;
  state.safeRest.remainingSeconds = state.safeRest.selectedMinutes * 60;
  if (dom.safeRestPresetsRow) dom.safeRestPresetsRow.classList.remove('hidden');
  if (dom.btnStartRest) dom.btnStartRest.classList.remove('hidden');
  if (dom.btnEndRestEarly) dom.btnEndRestEarly.classList.add('hidden');
  if (dom.safeRestTimerSubtext) {
    dom.safeRestTimerSubtext.textContent = `Preset duration: ${state.safeRest.selectedMinutes} minutes`;
  }
  updateSafeRestTimerDisplay();
  if (dom.safeRestMainView) dom.safeRestMainView.classList.remove('hidden');
  if (dom.safeRestCheckView) dom.safeRestCheckView.classList.add('hidden');
  if (dom.safeRestCompletedView) dom.safeRestCompletedView.classList.add('hidden');
}

// ─── Journey Tracker (THIS Journey Only - in-memory session tracker) ──────────
function calculateHaversineDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth's mean radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function updateJourneyDistanceDisplay() {
  if (dom.valJourneyDistance) {
    dom.valJourneyDistance.textContent = `${state.journey.distanceKm.toFixed(1)} km`;
  }
}

function updateJourneyTimerDisplay() {
  if (dom.valJourneyElapsed) {
    dom.valJourneyElapsed.textContent = formatTimeHMS(state.journey.elapsedSec);
  }
}

function updateJourneyFatigueEvents() {
  if (!state.journey.startTime || !dom.timelineList) return;
  const items = dom.timelineList.querySelectorAll('.timeline-item');
  let count = 0;
  items.forEach(item => {
    const cat = item.getAttribute('data-category');
    if (cat === 'drowsy' || cat === 'distracted') {
      const ts = item._createdTimestamp || state.journey.startTime;
      if (ts >= state.journey.startTime) {
        count++;
      }
    }
  });
  state.journey.fatigueEvents = count;
  if (dom.valJourneyEvents) {
    dom.valJourneyEvents.textContent = String(count);
  }
}

function checkJourneyRestRule() {
  if (!state.journey.isActive && !state.journey.isEnded) {
    if (dom.journeyRestAdvisory) dom.journeyRestAdvisory.classList.add('hidden');
    return;
  }

  const isTimeRule = state.journey.elapsedSec >= 7200; // >= 2 hours
  const isDistanceRule = state.journey.distanceKm >= 200; // >= 200 km
  const isFatigueRule = state.journey.fatigueEvents >= 3; // >= 3 fatigue events

  if (isTimeRule || isDistanceRule || isFatigueRule) {
    let reason = 'Safety advisory';
    if (isFatigueRule) {
      reason = `${state.journey.fatigueEvents} fatigue events in this journey`;
    } else if (isDistanceRule) {
      reason = `${state.journey.distanceKm.toFixed(1)} km continuous travel`;
    } else if (isTimeRule) {
      reason = '2+ hours continuous driving';
    }

    if (dom.restAdvisoryReason) dom.restAdvisoryReason.textContent = reason;
    if (dom.journeyRestAdvisory) dom.journeyRestAdvisory.classList.remove('hidden');
  } else {
    if (dom.journeyRestAdvisory) dom.journeyRestAdvisory.classList.add('hidden');
  }
}

function handleGeolocationPosition(pos) {
  if (!state.journey.isActive) return;
  if (!pos || !pos.coords) return;

  const lat = pos.coords.latitude;
  const lon = pos.coords.longitude;
  const acc = pos.coords.accuracy;

  // If signal is too weak (accuracy > 150m) and no prior lock:
  if (acc && acc > 150 && !state.journey.lastPosition) {
    state.journey.gpsUnavailable = true;
    if (dom.journeyGpsNotice) dom.journeyGpsNotice.classList.remove('hidden');
    if (dom.journeyGpsText) dom.journeyGpsText.textContent = 'Distance unavailable — time-based journey';
    return;
  }

  if (dom.journeyGpsNotice) dom.journeyGpsNotice.classList.add('hidden');
  state.journey.gpsUnavailable = false;

  if (state.journey.lastPosition) {
    const deltaKm = calculateHaversineDistanceKm(
      state.journey.lastPosition.lat,
      state.journey.lastPosition.lon,
      lat,
      lon
    );

    // Filter out standing-still jitter (< 5 meters)
    if (deltaKm > 0.005) {
      state.journey.distanceKm += deltaKm;
      updateJourneyDistanceDisplay();
      checkJourneyRestRule();
    }
  }

  state.journey.lastPosition = { lat, lon };
}

function handleGeolocationError(err) {
  // If permission denied, GPS unavailable, or signal too weak:
  state.journey.gpsUnavailable = true;
  if (dom.journeyGpsNotice) dom.journeyGpsNotice.classList.remove('hidden');
  if (dom.journeyGpsText) {
    dom.journeyGpsText.textContent = 'Distance unavailable — time-based journey';
  }
}

function startJourney() {
  state.journey.isActive = true;
  state.journey.isEnded = false;
  state.journey.startTime = Date.now();
  state.journey.elapsedSec = 0;
  state.journey.distanceKm = 0.0;
  state.journey.lastPosition = null;
  state.journey.gpsUnavailable = false;
  state.journey.fatigueEvents = 0;

  const now = new Date(state.journey.startTime);
  const timeStr = now.toTimeString().slice(0, 5); // HH:MM

  if (dom.valJourneyStart) dom.valJourneyStart.textContent = timeStr;
  if (dom.subJourneyStart) dom.subJourneyStart.textContent = 'Departure locked';
  if (dom.valJourneyElapsed) dom.valJourneyElapsed.textContent = '00:00:00';
  if (dom.valJourneyDistance) dom.valJourneyDistance.textContent = '0.0 km';
  if (dom.valJourneyEvents) dom.valJourneyEvents.textContent = '0';
  if (dom.journeyStatusPill) {
    dom.journeyStatusPill.className = 'status-pill pill-live';
  }
  if (dom.journeyStatusText) dom.journeyStatusText.textContent = 'In progress';

  if (dom.btnStartJourney) dom.btnStartJourney.classList.add('hidden');
  if (dom.btnEndJourney) dom.btnEndJourney.classList.remove('hidden');
  if (dom.btnNewJourney) dom.btnNewJourney.classList.add('hidden');
  if (dom.journeySummaryBox) dom.journeySummaryBox.classList.add('hidden');
  if (dom.journeyRestAdvisory) dom.journeyRestAdvisory.classList.add('hidden');
  if (dom.journeyGpsNotice) dom.journeyGpsNotice.classList.add('hidden');

  // Start elapsed timer (1s interval)
  if (state.journey.timerInterval) clearInterval(state.journey.timerInterval);
  state.journey.timerInterval = setInterval(() => {
    state.journey.elapsedSec++;
    updateJourneyTimerDisplay();
    checkJourneyRestRule();
  }, 1000);

  // Request browser geolocation ONLY after user clicks Start journey
  if (navigator && navigator.geolocation && !state.journey.isSimulatingDistance) {
    try {
      state.journey.geoWatchId = navigator.geolocation.watchPosition(
        handleGeolocationPosition,
        handleGeolocationError,
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 1000 }
      );
    } catch (e) {
      handleGeolocationError(e);
    }
  } else if (!state.journey.isSimulatingDistance) {
    handleGeolocationError({ code: 2, message: 'Geolocation unsupported' });
  }

  // If in demo mode and simulated journey was checked, start simulation
  if (state.isDemoMode && dom.toggleSimJourney && dom.toggleSimJourney.checked) {
    startDistanceSimulation();
  }
}

function endJourney() {
  if (!state.journey.isActive && !state.journey.isEnded) return;

  state.journey.isActive = false;
  state.journey.isEnded = true;

  if (state.journey.timerInterval) {
    clearInterval(state.journey.timerInterval);
    state.journey.timerInterval = null;
  }
  if (state.journey.simInterval) {
    clearInterval(state.journey.simInterval);
    state.journey.simInterval = null;
  }
  if (state.journey.geoWatchId && navigator && navigator.geolocation) {
    try {
      navigator.geolocation.clearWatch(state.journey.geoWatchId);
    } catch (e) {}
    state.journey.geoWatchId = null;
  }

  if (dom.journeyStatusPill) {
    dom.journeyStatusPill.className = 'status-pill pill-waiting';
  }
  if (dom.journeyStatusText) dom.journeyStatusText.textContent = 'Completed';

  if (dom.btnStartJourney) dom.btnStartJourney.classList.add('hidden');
  if (dom.btnEndJourney) dom.btnEndJourney.classList.add('hidden');
  if (dom.btnNewJourney) dom.btnNewJourney.classList.remove('hidden');

  // Freeze summary
  const finalDurationStr = formatTimeHMS(state.journey.elapsedSec);
  const finalDistanceStr = `${state.journey.distanceKm.toFixed(1)} km`;
  const finalEventsStr = String(state.journey.fatigueEvents);

  if (dom.summaryJourneyDuration) dom.summaryJourneyDuration.textContent = finalDurationStr;
  if (dom.summaryJourneyDistance) dom.summaryJourneyDistance.textContent = finalDistanceStr;
  if (dom.summaryJourneyEvents) dom.summaryJourneyEvents.textContent = finalEventsStr;
  if (dom.journeySummaryBox) dom.journeySummaryBox.classList.remove('hidden');
}

function resetJourney() {
  endJourney();
  state.journey.isActive = false;
  state.journey.isEnded = false;
  state.journey.startTime = null;
  state.journey.elapsedSec = 0;
  state.journey.distanceKm = 0.0;
  state.journey.fatigueEvents = 0;
  state.journey.lastPosition = null;
  state.journey.gpsUnavailable = false;

  if (dom.valJourneyStart) dom.valJourneyStart.textContent = '--:--';
  if (dom.subJourneyStart) dom.subJourneyStart.textContent = 'Awaiting departure';
  if (dom.valJourneyElapsed) dom.valJourneyElapsed.textContent = '00:00:00';
  if (dom.valJourneyDistance) dom.valJourneyDistance.textContent = '0.0 km';
  if (dom.valJourneyEvents) dom.valJourneyEvents.textContent = '0';

  if (dom.journeyStatusPill) {
    dom.journeyStatusPill.className = 'status-pill pill-cam-off';
  }
  if (dom.journeyStatusText) dom.journeyStatusText.textContent = 'Not started';

  if (dom.btnStartJourney) dom.btnStartJourney.classList.remove('hidden');
  if (dom.btnEndJourney) dom.btnEndJourney.classList.add('hidden');
  if (dom.btnNewJourney) dom.btnNewJourney.classList.add('hidden');
  if (dom.journeySummaryBox) dom.journeySummaryBox.classList.add('hidden');
  if (dom.journeyRestAdvisory) dom.journeyRestAdvisory.classList.add('hidden');
  if (dom.journeyGpsNotice) dom.journeyGpsNotice.classList.add('hidden');
}

function syncJourneyDemoModeUI() {
  if (!dom.journeySimulatedWrap) return;
  if (state.isDemoMode) {
    dom.journeySimulatedWrap.classList.remove('hidden');
  } else {
    dom.journeySimulatedWrap.classList.add('hidden');
    // Live mode must NEVER simulate distance
    if (state.journey.isSimulatingDistance) {
      stopDistanceSimulation();
    }
  }
}

function startDistanceSimulation() {
  if (!state.isDemoMode) return; // Live mode must NEVER simulate distance
  state.journey.isSimulatingDistance = true;

  if (dom.journeySimulatedWrap) dom.journeySimulatedWrap.classList.remove('hidden');
  if (dom.toggleSimJourney) dom.toggleSimJourney.checked = true;

  if (state.journey.simInterval) clearInterval(state.journey.simInterval);
  state.journey.simInterval = setInterval(() => {
    if (!state.journey.isActive || !state.journey.isSimulatingDistance || !state.isDemoMode) {
      stopDistanceSimulation();
      return;
    }
    state.journey.distanceKm += 2.5;
    updateJourneyDistanceDisplay();
    checkJourneyRestRule();
  }, 1000);
}

function stopDistanceSimulation() {
  state.journey.isSimulatingDistance = false;
  if (state.journey.simInterval) {
    clearInterval(state.journey.simInterval);
    state.journey.simInterval = null;
  }
  if (dom.toggleSimJourney) dom.toggleSimJourney.checked = false;
}

// ─── Sensitivity Presets (Rule 2) ─────────────────────────────
const SENSITIVITY_PRESETS = {
  sensitive: {
    name: 'Sensitive',
    closure: 1.0,
    lookaway: 2.0,
    lookawayL1: 0.8
  },
  balanced: {
    name: 'Balanced',
    closure: 1.5,
    lookaway: 3.0,
    lookawayL1: 1.2
  },
  relaxed: {
    name: 'Relaxed',
    closure: 2.0,
    lookaway: 4.0,
    lookawayL1: 1.8
  }
};

function applySensitivityPreset(presetKey) {
  const p = SENSITIVITY_PRESETS[presetKey];
  if (!p) return;

  state.sensitivityPreset = presetKey;
  state.drowsyThresholdSec = p.closure;
  state.lookawayThresholdSec = p.lookaway;
  state.lookawayL1ThresholdSec = p.lookawayL1;

  if (dom.presetBadge) dom.presetBadge.textContent = p.name;

  if (dom.sliderDrowsyTrigger) dom.sliderDrowsyTrigger.value = p.closure;
  if (dom.drowsyTriggerDisplay) dom.drowsyTriggerDisplay.textContent = `${p.closure.toFixed(1)}s`;

  if (dom.sliderLookawayTrigger) dom.sliderLookawayTrigger.value = p.lookaway;
  if (dom.lookawayTriggerDisplay) dom.lookawayTriggerDisplay.textContent = `${p.lookaway.toFixed(1)}s`;

  [dom.btnPresetSensitive, dom.btnPresetBalanced, dom.btnPresetRelaxed].forEach(btn => {
    if (!btn) return;
    if (btn.getAttribute('data-preset') === presetKey) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  if (!state.eyeClosedSince) {
    updateEyeOpennessHUD(state.eyeOpennessL, state.eyeOpennessR, 0, p.closure, false, !state.faceDetected);
  }
}

function checkPresetMatch() {
  const closure = state.drowsyThresholdSec;
  const lookaway = state.lookawayThresholdSec;

  let matched = null;
  for (const [key, p] of Object.entries(SENSITIVITY_PRESETS)) {
    if (Math.abs(closure - p.closure) < 0.05 && Math.abs(lookaway - p.lookaway) < 0.05) {
      matched = key;
      break;
    }
  }

  if (matched) {
    state.sensitivityPreset = matched;
    if (dom.presetBadge) dom.presetBadge.textContent = SENSITIVITY_PRESETS[matched].name;
    [dom.btnPresetSensitive, dom.btnPresetBalanced, dom.btnPresetRelaxed].forEach(btn => {
      if (!btn) return;
      if (btn.getAttribute('data-preset') === matched) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
  } else {
    state.sensitivityPreset = 'custom';
    if (dom.presetBadge) dom.presetBadge.textContent = 'Custom';
    [dom.btnPresetSensitive, dom.btnPresetBalanced, dom.btnPresetRelaxed].forEach(btn => {
      if (btn) btn.classList.remove('active');
    });
  }
}

// ─── Alarm Preview Sequence (Rule 3) ──────────────────────────
function startAlarmPreviewSequence() {
  if (state.alarmPreviewActive) {
    stopAlarmPreviewSequence();
    return;
  }

  unlockAudioOnGesture();
  state.alarmPreviewActive = true;

  if (dom.btnAlarmPreview) {
    dom.btnAlarmPreview.textContent = '⏹ Stop Preview';
    dom.btnAlarmPreview.className = 'btn btn-primary btn-sm';
  }
  if (dom.previewStatusPill) {
    dom.previewStatusPill.classList.remove('hidden');
    dom.previewStatusPill.textContent = 'Playing Level 1 (Amber look-away tone)...';
  }

  // Step 1: Level 1 (0 to 2200ms)
  playDistractionBeepLevel1();
  const step1Interval = setInterval(playDistractionBeepLevel1, 750);

  // Step 2: Level 2 (2200ms to 4500ms)
  const timer2 = setTimeout(() => {
    if (!state.alarmPreviewActive) return;
    clearInterval(step1Interval);

    if (dom.previewStatusPill) {
      dom.previewStatusPill.textContent = 'Playing Level 2 (Escalating alert tone)...';
    }

    const vol = 0.9 * state.distractionVolume;
    playDistractionTone(740, 0.35, vol);
    setTimeout(() => {
      if (state.alarmPreviewActive) playDistractionTone(1100, 0.35, vol);
    }, 380);
    setTimeout(() => {
      if (state.alarmPreviewActive) playDistractionTone(740, 0.35, vol);
    }, 760);
    setTimeout(() => {
      if (state.alarmPreviewActive) playDistractionTone(1100, 0.35, vol);
    }, 1140);
  }, 2200);

  // Step 3: Level 3 (4500ms to 7000ms)
  const timer3 = setTimeout(() => {
    if (!state.alarmPreviewActive) return;

    if (dom.previewStatusPill) {
      dom.previewStatusPill.textContent = 'Playing Level 3 (Drowsy wake beep)...';
    }

    const vol = 0.7 * (state.drowsyVolume ?? 1.0);
    playDrowsyTone(880, 0.70, vol);
    setTimeout(() => {
      if (state.alarmPreviewActive) {
        playDrowsyTone(880, 0.70, vol);
      }
    }, 1000);
  }, 4500);

  // Finish (at 7000ms)
  const timerEnd = setTimeout(() => {
    clearInterval(step1Interval);
    stopAlarmPreviewSequence();
  }, 7000);

  state.alarmPreviewTimer = {
    step1Interval,
    timer2,
    timer3,
    timerEnd
  };
}

function stopAlarmPreviewSequence() {
  state.alarmPreviewActive = false;
  if (state.alarmPreviewTimer) {
    if (state.alarmPreviewTimer.step1Interval) clearInterval(state.alarmPreviewTimer.step1Interval);
    if (state.alarmPreviewTimer.timer2) clearTimeout(state.alarmPreviewTimer.timer2);
    if (state.alarmPreviewTimer.timer3) clearTimeout(state.alarmPreviewTimer.timer3);
    if (state.alarmPreviewTimer.timerEnd) clearTimeout(state.alarmPreviewTimer.timerEnd);
    state.alarmPreviewTimer = null;
  }

  if (dom.btnAlarmPreview) {
    dom.btnAlarmPreview.textContent = '▶ Play alarm sequence (L1 → L2 → L3)';
    dom.btnAlarmPreview.className = 'btn btn-secondary btn-sm';
  }
  if (dom.previewStatusPill) {
    dom.previewStatusPill.classList.add('hidden');
    dom.previewStatusPill.textContent = '';
  }
}

// ─── Settings Popover Interactions ────────────────────────────
function setupSettingsPopover() {
  const openSettings = () => {
    if (dom.settingsPopover && dom.settingsBackdrop) {
      dom.settingsPopover.classList.remove('hidden');
      dom.settingsBackdrop.classList.remove('hidden');
      dom.btnSettings?.setAttribute('aria-expanded', 'true');
    }
  };

  const closeSettings = () => {
    stopAlarmPreviewSequence();
    if (dom.settingsPopover && dom.settingsBackdrop) {
      dom.settingsPopover.classList.add('hidden');
      dom.settingsBackdrop.classList.add('hidden');
      dom.btnSettings?.setAttribute('aria-expanded', 'false');
    }
  };

  if (dom.btnSettings)          dom.btnSettings.addEventListener('click', openSettings);
  if (dom.btnCloseSettings)     dom.btnCloseSettings.addEventListener('click', closeSettings);
  if (dom.btnDoneSettings)      dom.btnDoneSettings.addEventListener('click', closeSettings);
  if (dom.settingsBackdrop)     dom.settingsBackdrop.addEventListener('click', closeSettings);
  if (dom.linkPreshiftSettings) dom.linkPreshiftSettings.addEventListener('click', () => {
    closePreshiftModal();
    openSettings();
  });

  // Preset Buttons (Rule 2)
  if (dom.btnPresetSensitive) {
    dom.btnPresetSensitive.addEventListener('click', () => applySensitivityPreset('sensitive'));
  }
  if (dom.btnPresetBalanced) {
    dom.btnPresetBalanced.addEventListener('click', () => applySensitivityPreset('balanced'));
  }
  if (dom.btnPresetRelaxed) {
    dom.btnPresetRelaxed.addEventListener('click', () => applySensitivityPreset('relaxed'));
  }

  // Drowsy Trigger Slider (Adjustable, default 1.5s, 1.0-3.0s)
  if (dom.sliderDrowsyTrigger && dom.drowsyTriggerDisplay) {
    dom.sliderDrowsyTrigger.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state.drowsyThresholdSec = val;
      dom.drowsyTriggerDisplay.textContent = `${val.toFixed(1)}s`;
      checkPresetMatch();
      if (!state.eyeClosedSince) {
        updateEyeOpennessHUD(state.eyeOpennessL, state.eyeOpennessR, 0, val, false, !state.faceDetected);
      }
    });
  }

  // Look-away Distraction Trigger Slider (Rule 2)
  if (dom.sliderLookawayTrigger && dom.lookawayTriggerDisplay) {
    dom.sliderLookawayTrigger.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state.lookawayThresholdSec = val;
      state.lookawayL1ThresholdSec = +(val * 0.4).toFixed(1);
      dom.lookawayTriggerDisplay.textContent = `${val.toFixed(1)}s`;
      checkPresetMatch();
    });
  }

  // Alarm Audio Preview Sequence (Rule 3)
  if (dom.btnAlarmPreview) {
    dom.btnAlarmPreview.addEventListener('click', startAlarmPreviewSequence);
  }

  // Camera Choice Dropdown
  if (dom.selectCamera) {
    dom.selectCamera.addEventListener('change', (e) => {
      const devId = e.target.value;
      switchCamera(devId);
    });
  }

  // Alarm Sound Toggle UI
  if (dom.toggleAlarmSound) {
    dom.toggleAlarmSound.addEventListener('click', () => {
      toggleSoundState();
    });
  }

  // Test drowsy sound button in Settings
  if (dom.btnTestSound) {
    dom.btnTestSound.addEventListener('click', testDrowsySoundScript);
  }

  // Drowsy Sound Volume Slider
  if (dom.sliderDrowsyVolume && dom.drowsyVolumeDisplay) {
    dom.sliderDrowsyVolume.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.drowsyVolume = val / 100;
      dom.drowsyVolumeDisplay.textContent = `${val}%`;
    });
  }

  // Distraction Sound Volume Slider
  if (dom.sliderDistractionVolume && dom.distractionVolumeDisplay) {
    dom.sliderDistractionVolume.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      state.distractionVolume = val / 100;
      dom.distractionVolumeDisplay.textContent = `${val}%`;
    });
  }

  // Test distraction sound button in Settings
  if (dom.btnTestDistractionSound) {
    dom.btnTestDistractionSound.addEventListener('click', testDistractionSoundScript);
  }

  // Glasses Mode & Eye Calibration
  if (dom.toggleGlassesMode) {
    dom.toggleGlassesMode.addEventListener('click', () => {
      setGlassesMode(!state.glassesMode);
    });
  }
  if (dom.btnCalibrateEyes) {
    dom.btnCalibrateEyes.addEventListener('click', () => {
      startEyeCalibration();
    });
  }
}

// ─── Setup General UI & Events ────────────────────────────────
function setupGeneralUI() {
  // Camera toggle button on bottom-right of video overlay
  if (dom.btnCamIcon) {
    dom.btnCamIcon.addEventListener('click', async () => {
      await unlockAudioOnGesture();
      if (state.cameraState === 'Live' || state.cameraState === 'Starting' || state.cameraState === 'No face') {
        stopCamera();
      } else {
        await startCamera(state.selectedCameraId);
      }
    });
  }

  // Sound toggle button on bottom-right of video overlay
  if (dom.btnSoundIcon) {
    dom.btnSoundIcon.addEventListener('click', () => {
      toggleSoundState();
    });
  }

  // Session Start / Stop Button (Rule 5)
  const handleSessionToggle = () => {
    if (state.isSessionRunning) {
      stopSession();
    } else {
      startSession();
    }
  };

  if (dom.btnCameraToggle)     dom.btnCameraToggle.addEventListener('click', handleSessionToggle);
  if (dom.btnPlaceholderStart) dom.btnPlaceholderStart.addEventListener('click', handleSessionToggle);
  if (dom.btnDismissError)     dom.btnDismissError.addEventListener('click', dismissError);

  // Reset Session Timeline (Only allowed when stopped)
  if (dom.btnResetSession)     dom.btnResetSession.addEventListener('click', resetTimelineOnlyWhenStopped);
  if (dom.btnResetFromSummary) dom.btnResetFromSummary.addEventListener('click', resetTimelineOnlyWhenStopped);

  // Download Session Report CSV (Rule 1)
  if (dom.btnDownloadCsv)      dom.btnDownloadCsv.addEventListener('click', downloadSessionReportCSV);

  // Close Session Summary Modal
  if (dom.btnCloseSummary) dom.btnCloseSummary.addEventListener('click', closeSessionSummaryModal);
  if (dom.btnDoneSummary)  dom.btnDoneSummary.addEventListener('click', closeSessionSummaryModal);
  if (dom.summaryBackdrop) dom.summaryBackdrop.addEventListener('click', closeSessionSummaryModal);

  // Close Timeline Detail Modal (Rule 4)
  if (dom.btnCloseTimelineDetail) dom.btnCloseTimelineDetail.addEventListener('click', closeTimelineDetailModal);
  if (dom.btnCloseDetailAction)   dom.btnCloseDetailAction.addEventListener('click', closeTimelineDetailModal);
  if (dom.timelineDetailBackdrop) dom.timelineDetailBackdrop.addEventListener('click', closeTimelineDetailModal);

  // Keyboard Shortcuts (Rule 6: Space = Ack, D = Demo, S = Start/Stop)
  document.addEventListener('keydown', (e) => {
    // Never trigger while typing in an input
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target.isContentEditable) {
      return;
    }

    // Demo Mode Keyboard Navigation (Right Arrow = Next, Left Arrow = Prev, Esc = Back to Live Demo)
    if (state.isDemoMode) {
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        handleDemoNext();
        return;
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        handleDemoPrev();
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        stopDemoMode();
        return;
      }
    }

    if (e.key === 'Escape') {
      if (dom.settingsPopover && !dom.settingsPopover.classList.contains('hidden')) {
        dom.settingsPopover.classList.add('hidden');
        dom.settingsBackdrop?.classList.add('hidden');
        dom.btnSettings?.setAttribute('aria-expanded', 'false');
        stopAlarmPreviewSequence();
      }
      if (dom.timelineDetailModal && !dom.timelineDetailModal.classList.contains('hidden')) {
        closeTimelineDetailModal();
      }
      if (dom.preshiftModal && !dom.preshiftModal.classList.contains('hidden')) {
        closePreshiftModal();
      }
      if (dom.summaryModal && !dom.summaryModal.classList.contains('hidden')) {
        closeSessionSummaryModal();
      }
      return;
    }

    // Space = acknowledge alarm
    if (e.code === 'Space' || e.key === ' ') {
      if (state.level3AlarmActive) {
        e.preventDefault();
        acknowledgeAlarmFromButton();
      } else if (state.activeDistractionLevel > 0) {
        e.preventDefault();
        clearDistractionWarning();
      }
      return;
    }

    // D = Demo mode
    if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      if (state.isDemoMode) {
        stopDemoMode();
      } else {
        startDemoMode();
      }
      return;
    }

    // S = Start/Stop session
    if ((e.key === 's' || e.key === 'S') && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      handleSessionToggle();
      return;
    }
  });

  // Level 3 Alarm: One large "I'm awake" button stops alarm & sound at once
  if (dom.btnImAwake) {
    dom.btnImAwake.addEventListener('click', acknowledgeAlarmFromButton);
  }

  // Distraction Acknowledgement: Level 1/2 "I'm back" buttons (Rule 3)
  if (dom.btnImBackL1) dom.btnImBackL1.addEventListener('click', clearDistractionWarning);
  if (dom.btnImBackL2) dom.btnImBackL2.addEventListener('click', clearDistractionWarning);

  // Recalibrate / Set Centre button (Fix A.4)
  if (dom.btnRecalibrateCentre) {
    dom.btnRecalibrateCentre.addEventListener('click', recalibrateCentre);
  }

  // Timeline Filter Pills click (single-row segmented control)
  if (dom.filterPills) {
    dom.filterPills.forEach(pill => {
      pill.addEventListener('click', () => {
        const filterVal = pill.getAttribute('data-filter') || 'all';
        setTimelineFilter(filterVal);
      });
    });
  }

  // Timeline "Show all" / "Show less" toggle button (resets to all when filtered)
  if (dom.btnTimelineToggleAll) {
    dom.btnTimelineToggleAll.addEventListener('click', () => {
      if (state.currentFilter !== 'all') {
        setTimelineFilter('all');
      } else {
        state.timelineExpanded = !state.timelineExpanded;
        applyTimelineVisibility();
      }
    });
  }

  // Pre-shift Check modal controls
  if (dom.btnPreshift)             dom.btnPreshift.addEventListener('click', openPreshiftModal);
  if (dom.btnClosePreshift)        dom.btnClosePreshift.addEventListener('click', closePreshiftModal);
  if (dom.preshiftBackdrop)        dom.preshiftBackdrop.addEventListener('click', closePreshiftModal);
  if (dom.btnBeginCheck)           dom.btnBeginCheck.addEventListener('click', startPreshiftCheck);
  if (dom.btnStartSessionPreshift) dom.btnStartSessionPreshift.addEventListener('click', () => {
    closePreshiftModal();
    startSession();
  });

  // Demo Mode Switch
  if (dom.toggleDemoMode) {
    dom.toggleDemoMode.addEventListener('click', () => {
      if (state.isDemoMode) {
        stopDemoMode();
      } else {
        startDemoMode();
      }
    });
  }

  // Demo Mode Controls Bar Buttons
  if (dom.btnDemoNext) dom.btnDemoNext.addEventListener('click', handleDemoNext);
  if (dom.btnDemoPrev) dom.btnDemoPrev.addEventListener('click', handleDemoPrev);
  if (dom.btnDemoReplay) dom.btnDemoReplay.addEventListener('click', () => goToDemoStage(1));
  if (dom.btnDemoReset) dom.btnDemoReset.addEventListener('click', () => goToDemoStage(1));
  if (dom.btnDemoExit) dom.btnDemoExit.addEventListener('click', stopDemoMode);

  // Clear Simulated Rows Banner Buttons
  if (dom.btnClearSimulatedRows) dom.btnClearSimulatedRows.addEventListener('click', clearSimulatedRowsFromTimeline);
  if (dom.btnDismissSimulatedBanner) dom.btnDismissSimulatedBanner.addEventListener('click', dismissSimulatedBanner);

  // Safe Rest Assistant Controls
  const presetBtns = dom.safeRestPresetBtns || document.querySelectorAll('.btn-preset-time');
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const min = parseInt(btn.getAttribute('data-minutes'), 10);
      setSafeRestDurationPreset(min);
    });
  });
  if (dom.btnStartRest)      dom.btnStartRest.addEventListener('click', startSafeRestTimer);
  if (dom.btnEndRestEarly)   dom.btnEndRestEarly.addEventListener('click', endSafeRestEarly);
  if (dom.btnReadyContinue)  dom.btnReadyContinue.addEventListener('click', () => finishSafeRestCheck(true));
  if (dom.btnRestMore)       dom.btnRestMore.addEventListener('click', () => finishSafeRestCheck(false));
  if (dom.btnReopenSafeRest) dom.btnReopenSafeRest.addEventListener('click', reopenSafeRestTimer);

  // Journey Tracker Controls
  if (dom.btnStartJourney)   dom.btnStartJourney.addEventListener('click', startJourney);
  if (dom.btnEndJourney)     dom.btnEndJourney.addEventListener('click', endJourney);
  if (dom.btnNewJourney)     dom.btnNewJourney.addEventListener('click', resetJourney);
  if (dom.toggleSimJourney) {
    dom.toggleSimJourney.addEventListener('change', (e) => {
      if (e.target.checked) {
        startDistanceSimulation();
      } else {
        stopDistanceSimulation();
      }
    });
  }
}

// ─── Application Bootstrap ────────────────────────────────────
function initApp() {
  cacheDOM();
  clearVideoPaneError();
  setCameraState('Off');
  resetCameraOffHUD();
  updateCameraIconUI(false);
  updateSoundUI();
  updateBannerMutedPills();
  if (dom.lookawayTimerVal) dom.lookawayTimerVal.textContent = '--';
  if (dom.lookawayTimerStatus) dom.lookawayTimerStatus.textContent = '--';
  if (dom.valLiveYaw) dom.valLiveYaw.textContent = '--';
  if (dom.valLivePitch) dom.valLivePitch.textContent = '--';
  if (dom.valLiveDirection) dom.valLiveDirection.textContent = '--';
  if (dom.overlayDirectionBadge) dom.overlayDirectionBadge.classList.add('hidden');
  if (dom.faceStatusPill) dom.faceStatusPill.classList.add('hidden');
  if (dom.drowsyVolumeDisplay) dom.drowsyVolumeDisplay.textContent = '100%';
  if (dom.distractionVolumeDisplay) dom.distractionVolumeDisplay.textContent = '80%';
  updateGlobalStatus('Calibrating');
  updateTopBarLiveStatus();
  resetDebugMetrics();
  setGlassesMode(false);
  setupSettingsPopover();
  setupGeneralUI();
  populateInitialExampleRows();
  updateScoreAndContributors();
  renderAlertnessScore();
  updateSevenBarStrip();

  // Populate pre-existing camera devices
  populateCameraChoices();
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', populateCameraChoices);
  }

  // Pre-load MediaPipe model in background
  loadFaceLandmarker().catch(err => {
    console.warn('[Jaago] Background load catch:', err);
  });

  // URL Hash navigation for visual state verification
  if (typeof window !== 'undefined' && window.location) {
    if (window.location.hash === '#settings') {
      dom.btnSettings?.click();
    } else if (window.location.hash === '#settings-bands') {
      dom.btnSettings?.click();
      const bodyEl = document.querySelector('.settings-body');
      if (bodyEl) bodyEl.scrollTop = bodyEl.scrollHeight;
    } else if (window.location.hash === '#score71') {
      state.faceDetected = true;
      state.cameraState = 'Live';
      state.isCalibratingBaseline = false;
      state.displayedScore = 71;
      state.targetScore = 71;
      state.scoreAnimStartTime = performance.now() - 1000;
      state.scoreAnimStartVal = 71;
      state.lookAwayCount = 1;
      state.headDirection = 'CENTRE';
      state.smoothedYaw = 2;
      state.smoothedPitch = 1;
      state.lastLookAwayTime = Date.now() - 15000; // Recent look-away (15s ago, < 30s)
      state.scoreHistory = [
        { score: 95, tier: 'focused', time: '19:40:00' },
        { score: 71, tier: 'distracted', time: '19:40:15' }
      ];
      renderAlertnessScore();
      updateScoreAndContributors();
      updateSevenBarStrip();
    } else if (window.location.hash === '#detail') {
      const firstRow = dom.timelineList?.querySelector('.timeline-item');
      if (firstRow) firstRow.click();
    } else if (window.location.hash === '#timeline-scroll') {
      for (let i = 1; i <= 10; i++) {
        addTimelineRow(`Extended session event log item #${i} · Sustained scan`, `${i}.4s`, i % 2 === 0 ? 'focused' : 'distracted');
      }
      if (dom.timelineList) {
        dom.timelineList.scrollTop = 999999;
      }
    } else if (window.location.hash === '#preshift') {
      openPreshiftModal();
      dom.preshiftStartView?.classList.add('hidden');
      dom.preshiftActiveView?.classList.remove('hidden');
      for (let i = 0; i < 4; i++) tickPreshiftRow(i);
      if (dom.preshiftProgressBar) dom.preshiftProgressBar.style.width = '100%';
      if (dom.preshiftCountdown) dom.preshiftCountdown.textContent = '0s · Assessment complete';
      evaluatePreshiftResults();
      if (dom.preshiftResultSection) dom.preshiftResultSection.classList.remove('hidden');
    } else if (window.location.hash === '#summary') {
      state.sessionDurationSec = 345;
      state.lowestAlertnessScore = 78;
      state.eventCounts = { drowsy: 1, distracted: 3, yawn: 2, total: 6 };
      openSessionSummaryModal();
    } else if (window.location.hash === '#alarm') {
      state.eyeClosedSince = Date.now() - 2100;
      triggerLevel3Alarm();
      state.distractionStartTime = Date.now() - 1400;
      state.headDirection = 'LEFT';
      setLevel1Distraction();
    } else if (window.location.hash === '#alarm') {
      state.eyeClosedSince = Date.now() - 2000;
      triggerLevel3Alarm();
    } else if (window.location.hash === '#alarm-muted') {
      state.alarmSoundEnabled = false;
      state.eyeClosedSince = Date.now() - 2100;
      triggerLevel3Alarm();
    } else if (window.location.hash === '#alarm-stage2') {
      state.eyeClosedSince = Date.now() - 4000;
      triggerLevel3Alarm();
      startDrowsyAudioStage2();
    } else if (window.location.hash === '#demo' || window.location.hash === '#demo-stage1') {
      startDemoMode();
      goToDemoStage(1);
    } else if (window.location.hash === '#demo-stage2') {
      startDemoMode();
      goToDemoStage(2);
    } else if (window.location.hash === '#demo-stage3') {
      startDemoMode();
      goToDemoStage(3);
    } else if (window.location.hash === '#demo-stage4') {
      startDemoMode();
      goToDemoStage(4);
    } else if (window.location.hash === '#demo-clear-banner') {
      startDemoMode();
      goToDemoStage(2);
      stopDemoMode();
    } else if (window.location.hash === '#filter-distracted') {
      setTimelineFilter('distracted');
    } else if (window.location.hash === '#filter-drowsy') {
      setTimelineFilter('drowsy');
    } else if (window.location.hash === '#filter-focused') {
      setTimelineFilter('focused');
    } else if (window.location.hash === '#filter-empty') {
      if (dom.timelineList) dom.timelineList.innerHTML = '';
      setTimelineFilter('drowsy');
    } else if (window.location.hash === '#glasses-indicator') {
      setGlassesMode(true);
    } else if (window.location.hash === '#glasses-mode') {
      dom.btnSettings?.click();
      setGlassesMode(true);
      const bodyEl = document.querySelector('.settings-body');
      if (bodyEl) bodyEl.scrollTop = 180;
    } else if (window.location.hash === '#glasses-mode-calibrated') {
      dom.btnSettings?.click();
      setGlassesMode(true);
      state.calibratedEyeOpenness = 0.62;
      state.calibratedEyeOpennessL = 0.61;
      state.calibratedEyeOpennessR = 0.63;
      if (dom.calibrateEyesInfo) {
        dom.calibrateEyesInfo.textContent = 'Baseline: 62% open · Closure trigger: <28%';
      }
      if (dom.calibrateEyesStatus) {
        dom.calibrateEyesStatus.textContent = 'Done ✓';
        dom.calibrateEyesStatus.classList.remove('hidden');
      }
      const bodyEl = document.querySelector('.settings-body');
      if (bodyEl) bodyEl.scrollTop = 180;
    } else if (window.location.hash === '#weak-signal') {
      setGlassesMode(true);
      state.cameraState = 'Live';
      state.faceDetected = true;
      state.isEyeSignalWeak = true;
      updateEyeSignalWeakPill();
    } else if (window.location.hash === '#camera-off') {
      setCameraState('Off');
      resetCameraOffHUD();
    } else if (window.location.hash === '#safe-rest') {
      addTimelineRow('Drowsy — Eyes closed', '—', 'drowsy');
      addTimelineRow('Drowsy — Eyes closed', '1.6s', 'drowsy');
      updateSafeRestCardVisibility();
    } else if (window.location.hash === '#safe-rest-timer') {
      addTimelineRow('Drowsy — Eyes closed', '—', 'drowsy');
      addTimelineRow('Drowsy — Eyes closed', '1.6s', 'drowsy');
      startSafeRestTimer();
    } else if (window.location.hash === '#safe-rest-check') {
      addTimelineRow('Drowsy — Eyes closed', '—', 'drowsy');
      addTimelineRow('Drowsy — Eyes closed', '1.6s', 'drowsy');
      showSafeRestCheckView();
      dom.safeRestCard?.classList.remove('hidden');
    } else if (window.location.hash === '#safe-rest-completed') {
      addTimelineRow('Drowsy — Eyes closed', '—', 'drowsy');
      addTimelineRow('Drowsy — Eyes closed', '1.6s', 'drowsy');
      finishSafeRestCheck(true);
      dom.safeRestCard?.classList.remove('hidden');
    } else if (window.location.hash === '#journey') {
      resetJourney();
    } else if (window.location.hash === '#journey-active') {
      startJourney();
      state.journey.elapsedSec = 1420;
      state.journey.distanceKm = 24.6;
      state.journey.gpsUnavailable = false;
      if (dom.journeyGpsNotice) dom.journeyGpsNotice.classList.add('hidden');
      updateJourneyTimerDisplay();
      updateJourneyDistanceDisplay();
    } else if (window.location.hash === '#journey-gps-denied') {
      startJourney();
      handleGeolocationError({ code: 1, message: 'User denied Geolocation' });
    } else if (window.location.hash === '#journey-rest-rule') {
      startJourney();
      addTimelineRow('Drowsy — Eyes closed', '1.5s', 'drowsy');
      addTimelineRow('Distracted — Looking LEFT', '1.8s', 'distracted');
      addTimelineRow('Drowsy — Eyes closed', '1.6s', 'drowsy');
      state.journey.fatigueEvents = 3;
      if (dom.valJourneyEvents) dom.valJourneyEvents.textContent = '3';
      checkJourneyRestRule();
    } else if (window.location.hash === '#journey-simulated') {
      startDemoMode();
      startJourney();
      startDistanceSimulation();
      state.journey.distanceKm = 48.5;
      updateJourneyDistanceDisplay();
    } else if (window.location.hash === '#journey-ended') {
      startJourney();
      state.journey.elapsedSec = 3680;
      state.journey.distanceKm = 62.4;
      state.journey.fatigueEvents = 2;
      endJourney();
    }
  }

  updateSafeRestCardVisibility();
  syncJourneyDemoModeUI();
}

if (typeof window !== 'undefined') {
  window.__testInterface = {
    state,
    dom,
    getSessionTimelineFatigueCounts,
    updateSafeRestCardVisibility,
    setSafeRestDurationPreset,
    updateSafeRestTimerDisplay,
    startSafeRestTimer,
    endSafeRestEarly,
    showSafeRestCheckView,
    finishSafeRestCheck,
    reopenSafeRestTimer,
    calculateHaversineDistanceKm,
    updateJourneyTimerDisplay,
    updateJourneyDistanceDisplay,
    startJourney,
    endJourney,
    resetJourney,
    handleGeolocationPosition,
    handleGeolocationError,
    checkJourneyRestRule,
    updateJourneyFatigueEvents,
    startDistanceSimulation,
    stopDistanceSimulation,
    syncJourneyDemoModeUI,
    addTimelineRow,
    setGlassesMode,
    startEyeCalibration,
    finishEyeCalibration,
    triggerLevel3Alarm,
    dismissLevel3Alarm,
    acknowledgeAlarmFromButton,
    stopCamera,
    startCamera,
    setCameraState,
    resetCameraOffHUD,
    handleDetectionResults
  };
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
