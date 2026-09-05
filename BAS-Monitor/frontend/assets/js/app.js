/**
 * BAS Experiment Monitor - Core Frontend Application Engine
 * ISRO SIH26174 Edge Human Activity Recognition & Sequence Validation
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Application State ---
    const state = {
        isAnalyzing: false,
        isPaused: false,
        isRecording: false,
        isStreaming: false,
        isVoiceEnabled: true,
        inputSource: 'standby', // 'standby', 'camera', 'file'
        activeVideoUrl: null,
        activeFileName: null,
        recordingStartTime: null,
        recordingTimerInterval: null,
        recSeconds: 0,
        currentStepIndex: 2, // 0-indexed (Step 3 active by default)
        confidence: 94,
        detectedAction: 'Handling Sample Container',
        validationState: 'VALID', // 'VALID', 'WARNING', 'ERROR'
        ipStreamUrl: 'rtsp://192.168.1.50:8554/live',
        analysisInterval: null,
        boxes: []
    };

    // --- Predefined Experiment Steps (ISRO Microgravity Experiment Sample) ---
    const experimentSteps = [
        {
            id: 1,
            title: "Container Retrieval & Setup",
            desc: "Retrieve test sample container from payload rack and verify seals.",
            expectedAction: "PICK_CONTAINER",
            durationEst: "30s"
        },
        {
            id: 2,
            title: "Sample Transfer & Pipetting",
            desc: "Transfer 5ml reagent into reaction vessel using automated pipette.",
            expectedAction: "PIPETTE_TRANSFER",
            durationEst: "45s"
        },
        {
            id: 3,
            title: "Analyzer Chamber Insertion",
            desc: "Insert reaction vessel firmly into optical analyzer slot B.",
            expectedAction: "INSERT_ANALYZER",
            durationEst: "20s"
        },
        {
            id: 4,
            title: "Optical & Telemetry Scan",
            desc: "Engage optical sensor probe and initiate 5-second spectroscopic read.",
            expectedAction: "INITIATE_SCAN",
            durationEst: "15s"
        },
        {
            id: 5,
            title: "Sealing & Storage",
            desc: "Cap reaction vessel, log telemetry batch, and return container to rack.",
            expectedAction: "SEAL_CONTAINER",
            durationEst: "25s"
        }
    ];

    // --- DOM Elements ---
    const elements = {
        currentTime: document.getElementById('current-time'),
        videoElement: document.getElementById('video-element'),
        aiCanvas: document.getElementById('ai-canvas'),
        videoPlaceholder: document.getElementById('video-placeholder'),
        videoSourceText: document.getElementById('video-source-text'),
        recStatusBadge: document.getElementById('rec-status-badge'),
        recTimerDisplay: document.getElementById('rec-timer-display'),
        aiDetectionOverlay: document.getElementById('ai-detection-overlay'),
        aiActionText: document.getElementById('ai-action-text'),
        
        // Procedure
        procedureStepsList: document.getElementById('procedure-steps-list'),
        
        // Status
        statusCurrentStep: document.getElementById('status-current-step'),
        validationBadge: document.getElementById('validation-badge'),
        confidenceVal: document.getElementById('confidence-val'),
        confidenceBar: document.getElementById('confidence-bar'),
        nextStepTitle: document.getElementById('next-step-title'),
        nextStepDesc: document.getElementById('next-step-desc'),
        
        // Diagnostics
        diagModel: document.getElementById('diag-model'),
        diagCamera: document.getElementById('diag-camera'),
        diagStorage: document.getElementById('diag-storage'),
        diagStream: document.getElementById('diag-stream'),
        
        // Terminal
        terminalLogFeed: document.getElementById('terminal-log-feed'),
        terminalInput: document.getElementById('terminal-input'),
        btnClearLog: document.getElementById('btn-clear-log'),
        
        // Controls
        btnSourceCam: document.getElementById('btn-source-cam'),
        btnSourceFile: document.getElementById('btn-source-file'),
        fileInput: document.getElementById('file-input'),
        btnToggleRec: document.getElementById('btn-toggle-rec'),
        btnToggleStream: document.getElementById('btn-toggle-stream'),
        btnToggleVoice: document.getElementById('btn-toggle-voice'),
        
        btnStartAnalysis: document.getElementById('btn-start-analysis'),
        btnPauseAnalysis: document.getElementById('btn-pause-analysis'),
        btnStopAnalysis: document.getElementById('btn-stop-analysis'),
        btnResetSeq: document.getElementById('btn-reset-seq'),
        
        btnStandbyCam: document.getElementById('btn-standby-cam'),
        btnStandbyUpload: document.getElementById('btn-standby-upload'),
        
        outputActiveFilename: document.getElementById('output-active-filename'),
        streamInfoText: document.getElementById('stream-info-text')
    };

    // --- Clock Engine ---
    function updateClock() {
        const now = new Date();
        const hrs = String(now.getHours()).padStart(2, '0');
        const mins = String(now.getMinutes()).padStart(2, '0');
        const secs = String(now.getSeconds()).padStart(2, '0');
        if (elements.currentTime) {
            elements.currentTime.textContent = `${hrs}:${mins}:${secs}`;
        }
    }
    setInterval(updateClock, 1000);
    updateClock();

    // --- Terminal Logging Engine ---
    function log(message, type = 'SYS') {
        if (!elements.terminalLogFeed) return;
        const now = new Date();
        const timestamp = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
        
        const entry = document.createElement('div');
        entry.className = 'flex items-start font-mono leading-tight hover:bg-surface-container-high/40 px-1 py-0.5 rounded transition-colors';
        
        let typeColor = 'text-primary-fixed-dim';
        if (type === 'AI') typeColor = 'text-tertiary-fixed-dim';
        if (type === 'WARN') typeColor = 'text-secondary-fixed-dim';
        if (type === 'ERR') typeColor = 'text-error';
        if (type === 'STREAM') typeColor = 'text-primary';

        entry.innerHTML = `
            <span class="text-on-surface-variant/60 mr-2 shrink-0 text-[10px]">${timestamp}</span>
            <span class="${typeColor} font-semibold mr-1.5 shrink-0">[${type}]</span>
            <span class="text-on-surface/90 flex-1 break-all">${escapeHtml(message)}</span>
        `;

        elements.terminalLogFeed.appendChild(entry);
        elements.terminalLogFeed.scrollTop = elements.terminalLogFeed.scrollHeight;
    }

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Initial Logs
    log("BAS Experiment Monitor initializing...", "SYS");
    log("Edge HAR Neural Network Model loaded (YOLOv8 + SlowFast).", "SYS");
    log("Sequence Engine active. Predefined Experiment: EXP-01 Loaded.", "SYS");
    log("System Ready. Waiting for camera input or video source.", "SYS");

    // Clear log button
    if (elements.btnClearLog) {
        elements.btnClearLog.addEventListener('click', () => {
            elements.terminalLogFeed.innerHTML = '';
            log("Terminal log buffer cleared.", "SYS");
        });
    }

    // Terminal Input parser
    if (elements.terminalInput) {
        elements.terminalInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const cmd = elements.terminalInput.value.trim().toLowerCase();
                elements.terminalInput.value = '';
                if (!cmd) return;
                
                log(`> ${cmd}`, "USER");
                handleTerminalCommand(cmd);
            }
        });
    }

    function handleTerminalCommand(cmd) {
        if (cmd === 'clear') {
            elements.terminalLogFeed.innerHTML = '';
            log("Terminal cleared.", "SYS");
        } else if (cmd === 'status') {
            log(`Status: Analyzing=${state.isAnalyzing}, Source=${state.inputSource}, Step=${state.currentStepIndex + 1}/${experimentSteps.length}, REC=${state.isRecording}, Stream=${state.isStreaming}`, "SYS");
        } else if (cmd === 'start') {
            startAnalysis();
        } else if (cmd === 'stop') {
            stopAnalysis();
        } else if (cmd === 'reset') {
            resetSequence();
        } else if (cmd === 'help') {
            log("Available commands: status, start, stop, reset, clear, help", "SYS");
        } else {
            log(`Unknown command: '${cmd}'. Type 'help' for available commands.`, "ERR");
        }
    }

    // --- Procedure Timeline Rendering ---
    function renderProcedureSteps() {
        if (!elements.procedureStepsList) return;
        elements.procedureStepsList.innerHTML = '';

        const container = document.createElement('div');
        container.className = 'relative space-y-4';

        // Connecting vertical line
        const line = document.createElement('div');
        line.className = 'absolute left-[15px] top-4 bottom-4 w-px bg-outline-variant/60 z-0';
        container.appendChild(line);

        experimentSteps.forEach((step, idx) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'flex items-start relative z-10 cursor-pointer group';
            
            const isCompleted = idx < state.currentStepIndex;
            const isCurrent = idx === state.currentStepIndex;
            const isPending = idx > state.currentStepIndex;

            let iconBg = 'bg-surface-container-high text-on-surface-variant border-outline-variant';
            let iconSymbol = 'radio_button_unchecked';
            let titleColor = 'text-on-surface-variant/70';
            let statusText = 'Pending';
            let statusColor = 'text-on-surface-variant/50';
            let cardExtra = '';

            if (isCompleted) {
                iconBg = 'bg-tertiary-container/30 text-tertiary-fixed border-tertiary-fixed/50';
                iconSymbol = 'check';
                titleColor = 'text-on-surface';
                statusText = 'Completed';
                statusColor = 'text-tertiary-fixed-dim';
            } else if (isCurrent) {
                iconBg = 'bg-primary-container/30 text-primary-fixed-dim border-2 border-primary-fixed-dim shadow-[0_0_12px_rgba(0,218,243,0.3)]';
                iconSymbol = 'adjust';
                titleColor = 'text-primary-fixed-dim font-bold';
                statusText = 'In Progress';
                statusColor = 'text-primary-fixed-dim';
                cardExtra = 'bg-surface-container-high/60 border border-primary-fixed-dim/30 p-2.5 rounded-md';
            }

            stepDiv.innerHTML = `
                <div class="w-8 h-8 rounded-full ${iconBg} flex items-center justify-center shrink-0 border mr-3 transition-all">
                    <span class="material-symbols-outlined text-[16px]">${iconSymbol}</span>
                </div>
                <div class="pt-0.5 flex-1 ${cardExtra}">
                    <div class="flex items-center justify-between">
                        <p class="font-data-lg text-xs ${titleColor}">Step ${step.id}: ${step.title}</p>
                        <span class="font-label-caps text-[10px] ${statusColor}">${statusText}</span>
                    </div>
                    <p class="font-caption text-on-surface-variant/80 text-[11px] mt-0.5 line-clamp-2">${step.desc}</p>
                </div>
            `;

            stepDiv.addEventListener('click', () => {
                state.currentStepIndex = idx;
                updateStatusUI();
                renderProcedureSteps();
                log(`Manual step selection: Step ${step.id} (${step.title})`, "SYS");
                speakVoice(`Step ${step.id}: ${step.title}`);
            });

            container.appendChild(stepDiv);
        });

        elements.procedureStepsList.appendChild(container);
    }

    // --- Status UI Update ---
    function updateStatusUI() {
        const currentStep = experimentSteps[state.currentStepIndex] || experimentSteps[0];
        const nextStep = experimentSteps[state.currentStepIndex + 1];

        if (elements.statusCurrentStep) {
            elements.statusCurrentStep.textContent = `Step ${currentStep.id}: ${currentStep.title}`;
        }

        if (elements.confidenceVal) {
            elements.confidenceVal.textContent = `${state.confidence}%`;
        }
        if (elements.confidenceBar) {
            elements.confidenceBar.style.width = `${state.confidence}%`;
        }

        if (elements.validationBadge) {
            if (state.validationState === 'VALID') {
                elements.validationBadge.className = 'bg-tertiary-container/15 text-tertiary-fixed border border-tertiary-fixed/30 font-label-caps text-[11px] px-2 py-0.5 rounded flex items-center';
                elements.validationBadge.innerHTML = `<span class="material-symbols-outlined text-[12px] mr-1">check_circle</span> VALID`;
            } else if (state.validationState === 'WARNING') {
                elements.validationBadge.className = 'bg-secondary-container/20 text-secondary-fixed border border-secondary-fixed/40 font-label-caps text-[11px] px-2 py-0.5 rounded flex items-center';
                elements.validationBadge.innerHTML = `<span class="material-symbols-outlined text-[12px] mr-1">warning</span> OUT OF ORDER`;
            } else {
                elements.validationBadge.className = 'bg-error-container/30 text-error border border-error/40 font-label-caps text-[11px] px-2 py-0.5 rounded flex items-center';
                elements.validationBadge.innerHTML = `<span class="material-symbols-outlined text-[12px] mr-1">error</span> SKIPPED STEP`;
            }
        }

        if (elements.nextStepTitle && elements.nextStepDesc) {
            if (nextStep) {
                elements.nextStepTitle.textContent = `Step ${nextStep.id}: ${nextStep.title}`;
                elements.nextStepDesc.textContent = nextStep.desc;
            } else {
                elements.nextStepTitle.textContent = "Procedure Complete";
                elements.nextStepDesc.textContent = "All experiment steps have been successfully validated.";
            }
        }

        // Diagnostics
        if (elements.diagCamera) {
            if (state.inputSource === 'camera') {
                elements.diagCamera.className = 'text-tertiary-fixed-dim flex items-center';
                elements.diagCamera.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-tertiary-fixed-dim mr-1.5"></span> WEBCAM LIVE';
            } else if (state.inputSource === 'file') {
                elements.diagCamera.className = 'text-primary-fixed-dim flex items-center';
                elements.diagCamera.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-primary-fixed-dim mr-1.5"></span> FILE FEED';
            } else {
                elements.diagCamera.className = 'text-on-surface-variant flex items-center';
                elements.diagCamera.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-outline-variant mr-1.5"></span> STANDBY';
            }
        }

        if (elements.diagStream) {
            if (state.isStreaming) {
                elements.diagStream.className = 'text-tertiary-fixed-dim flex items-center';
                elements.diagStream.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-tertiary-fixed-dim mr-1.5 animate-pulse"></span> ACTIVE';
            } else {
                elements.diagStream.className = 'text-on-surface-variant flex items-center';
                elements.diagStream.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-outline-variant mr-1.5"></span> OFF';
            }
        }
    }

    // --- Input Source Handlers ---
    function setSourceStandby() {
        stopVideoTrack();
        state.inputSource = 'standby';
        if (elements.videoElement) elements.videoElement.classList.add('hidden');
        if (elements.videoPlaceholder) elements.videoPlaceholder.classList.remove('hidden');
        if (elements.videoSourceText) elements.videoSourceText.textContent = "Source: Standby";
        if (elements.outputActiveFilename) elements.outputActiveFilename.textContent = "Source: None Selected";
        
        elements.btnSourceCam.className = "flex items-center px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-container-high font-label-caps text-xs transition-colors";
        elements.btnSourceFile.className = "flex items-center px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-container-high font-label-caps text-xs transition-colors";
        
        updateStatusUI();
    }

    async function startWebcam() {
        try {
            stopVideoTrack();
            log("Requesting access to webcam...", "SYS");
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
            
            if (elements.videoElement) {
                elements.videoElement.srcObject = stream;
                elements.videoElement.classList.remove('hidden');
                elements.videoElement.play();
            }
            if (elements.videoPlaceholder) elements.videoPlaceholder.classList.add('hidden');
            
            state.inputSource = 'camera';
            if (elements.videoSourceText) elements.videoSourceText.textContent = "Source: Live Webcam";
            if (elements.outputActiveFilename) elements.outputActiveFilename.textContent = "Source: Live Webcam Feed";
            
            elements.btnSourceCam.className = "flex items-center px-3 py-1.5 rounded border border-primary-fixed-dim text-primary-fixed-dim bg-primary-fixed-dim/10 font-label-caps text-xs transition-colors";
            elements.btnSourceFile.className = "flex items-center px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-container-high font-label-caps text-xs transition-colors";

            log("Live Webcam stream connected successfully.", "SYS");
            updateStatusUI();
        } catch (err) {
            log(`Failed to open webcam: ${err.message}. (Falling back to simulated stream)`, "ERR");
            setSimulatedVideoSource("Webcam Simulation");
        }
    }

    function setSimulatedVideoSource(name) {
        stopVideoTrack();
        state.inputSource = 'file';
        if (elements.videoPlaceholder) elements.videoPlaceholder.classList.add('hidden');
        if (elements.videoElement) elements.videoElement.classList.remove('hidden');
        
        if (elements.videoSourceText) elements.videoSourceText.textContent = `Source: ${name}`;
        if (elements.outputActiveFilename) elements.outputActiveFilename.textContent = `Source: ${name}`;

        elements.btnSourceCam.className = "flex items-center px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-container-high font-label-caps text-xs transition-colors";
        elements.btnSourceFile.className = "flex items-center px-3 py-1.5 rounded border border-primary-fixed-dim text-primary-fixed-dim bg-primary-fixed-dim/10 font-label-caps text-xs transition-colors";
        
        updateStatusUI();
    }

    function stopVideoTrack() {
        if (elements.videoElement && elements.videoElement.srcObject) {
            const tracks = elements.videoElement.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            elements.videoElement.srcObject = null;
        }
    }

    // Source buttons event listeners
    if (elements.btnSourceCam) elements.btnSourceCam.addEventListener('click', startWebcam);
    if (elements.btnStandbyCam) elements.btnStandbyCam.addEventListener('click', startWebcam);

    if (elements.btnSourceFile) {
        elements.btnSourceFile.addEventListener('click', () => elements.fileInput.click());
    }
    if (elements.btnStandbyUpload) {
        elements.btnStandbyUpload.addEventListener('click', () => elements.fileInput.click());
    }

    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                stopVideoTrack();
                const fileUrl = URL.createObjectURL(file);
                if (elements.videoElement) {
                    elements.videoElement.src = fileUrl;
                    elements.videoElement.classList.remove('hidden');
                    elements.videoElement.play();
                }
                if (elements.videoPlaceholder) elements.videoPlaceholder.classList.add('hidden');

                state.inputSource = 'file';
                state.activeFileName = file.name;
                
                if (elements.videoSourceText) elements.videoSourceText.textContent = `File: ${file.name}`;
                if (elements.outputActiveFilename) elements.outputActiveFilename.textContent = `File: ${file.name}`;

                elements.btnSourceCam.className = "flex items-center px-3 py-1.5 rounded border border-outline-variant text-on-surface hover:bg-surface-container-high font-label-caps text-xs transition-colors";
                elements.btnSourceFile.className = "flex items-center px-3 py-1.5 rounded border border-primary-fixed-dim text-primary-fixed-dim bg-primary-fixed-dim/10 font-label-caps text-xs transition-colors";

                log(`Video file loaded: ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`, "SYS");
                updateStatusUI();
            }
        });
    }

    // --- Output Services: Recording, Streaming, Voice ---
    if (elements.btnToggleRec) {
        elements.btnToggleRec.addEventListener('click', () => {
            state.isRecording = !state.isRecording;
            
            if (state.isRecording) {
                _showRecUI();
                startRecTimer();
                log("Local video recording started (.mp4 output stream).", "SYS");
                
                if (window.backend) {
                    window.backend.startRecording();
                }
            } else {
                _hideRecUI();
                stopRecTimer();
                log("Local video recording saved.", "SYS");
                
                if (window.backend) {
                    window.backend.stopRecording();
                }
            }
            updateStatusUI();
        });
    }

    // --- Python backend signal handlers (called from bridge init in index.html) ---

    // Expose log() so Python bridge signals can write to the terminal
    window.basLog = function(message, type) {
        log(message, type);
    };

    // Called every second by Python with "MM:SS" string
    window.basOnRecTimerTick = function(timeStr) {
        if (elements.recTimerDisplay) elements.recTimerDisplay.textContent = timeStr;
    };

    // Helper: show recording UI state
    function _showRecUI() {
        elements.btnToggleRec.className = "flex items-center px-3 py-1 rounded border border-error/50 bg-error-container/30 text-error font-label-caps text-xs transition-colors animate-pulse";
        elements.btnToggleRec.innerHTML = `<span class="w-2 h-2 rounded-full bg-error mr-1.5"></span> REC ON`;
        if (elements.recStatusBadge) elements.recStatusBadge.classList.remove('hidden');
    }

    // Helper: hide recording UI state
    function _hideRecUI() {
        elements.btnToggleRec.className = "flex items-center px-3 py-1 rounded border border-outline-variant text-on-surface-variant hover:text-on-surface font-label-caps text-xs transition-colors";
        elements.btnToggleRec.innerHTML = `<span class="w-2 h-2 rounded-full bg-outline-variant mr-1.5"></span> Record MP4`;
        if (elements.recStatusBadge) elements.recStatusBadge.classList.add('hidden');
    }

    function startRecTimer() {
        state.recSeconds = 0;
        clearInterval(state.recordingTimerInterval);
        state.recordingTimerInterval = setInterval(() => {
            state.recSeconds++;
            const mins = String(Math.floor(state.recSeconds / 60)).padStart(2, '0');
            const secs = String(state.recSeconds % 60).padStart(2, '0');
            if (elements.recTimerDisplay) elements.recTimerDisplay.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function stopRecTimer() {
        clearInterval(state.recordingTimerInterval);
    }

    if (elements.btnToggleStream) {
        elements.btnToggleStream.addEventListener('click', () => {
            state.isStreaming = !state.isStreaming;
            if (state.isStreaming) {
                elements.btnToggleStream.className = "flex items-center px-3 py-1 rounded border border-tertiary-fixed/40 bg-tertiary-container/20 text-tertiary-fixed font-label-caps text-xs transition-colors";
                elements.btnToggleStream.innerHTML = `<span class="material-symbols-outlined text-[14px] mr-1">podcasts</span> Streaming ON`;
                log(`RTSP IP Stream active at: ${state.ipStreamUrl}`, "STREAM");
            } else {
                elements.btnToggleStream.className = "flex items-center px-3 py-1 rounded border border-outline-variant text-on-surface-variant hover:text-on-surface font-label-caps text-xs transition-colors";
                elements.btnToggleStream.innerHTML = `<span class="material-symbols-outlined text-[14px] mr-1">podcasts</span> IP Stream`;
                log("RTSP IP Stream stopped.", "STREAM");
            }
            updateStatusUI();
        });
    }

    if (elements.btnToggleVoice) {
        elements.btnToggleVoice.addEventListener('click', () => {
            state.isVoiceEnabled = !state.isVoiceEnabled;
            if (state.isVoiceEnabled) {
                elements.btnToggleVoice.className = "flex items-center px-3 py-1 rounded border border-tertiary-fixed/30 bg-tertiary-container/10 text-tertiary-fixed-dim font-label-caps text-xs transition-colors";
                elements.btnToggleVoice.innerHTML = `<span class="material-symbols-outlined text-[14px] mr-1">volume_up</span> Voice ON`;
                log("Voice Guidance alerts enabled.", "SYS");
                speakVoice("Voice alerts enabled.");
            } else {
                elements.btnToggleVoice.className = "flex items-center px-3 py-1 rounded border border-outline-variant text-on-surface-variant hover:text-on-surface font-label-caps text-xs transition-colors";
                elements.btnToggleVoice.innerHTML = `<span class="material-symbols-outlined text-[14px] mr-1">volume_off</span> Voice MUTED`;
                log("Voice Guidance alerts muted.", "SYS");
            }
        });
    }

    function speakVoice(text) {
        if (!state.isVoiceEnabled || !('speechSynthesis' in window)) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
    }

    // --- Analysis Engine & Canvas Drawing ---
    function startAnalysis() {
        if (state.inputSource === 'standby') {
            log("Cannot start analysis: No video source selected.", "WARN");
            speakVoice("Please select a video input source.");
            return;
        }

        state.isAnalyzing = true;
        state.isPaused = false;
        
        if (elements.aiDetectionOverlay) elements.aiDetectionOverlay.classList.remove('hidden');
        if (elements.videoElement && elements.videoElement.paused) {
            elements.videoElement.play();
        }

        log(`Starting HAR Analysis pipeline on ${state.inputSource}...`, "AI");
        speakVoice(`Analysis started. Monitoring ${experimentSteps[state.currentStepIndex].title}`);

        startCanvasDrawing();
        startSimulationLoop();
    }

    function pauseAnalysis() {
        if (!state.isAnalyzing) return;
        state.isPaused = !state.isPaused;
        if (state.isPaused) {
            log("Analysis paused.", "AI");
            if (elements.videoElement) elements.videoElement.pause();
        } else {
            log("Analysis resumed.", "AI");
            if (elements.videoElement) elements.videoElement.play();
        }
    }

    function stopAnalysis() {
        state.isAnalyzing = false;
        state.isPaused = false;
        clearInterval(state.analysisInterval);
        
        if (elements.aiDetectionOverlay) elements.aiDetectionOverlay.classList.add('hidden');
        if (elements.videoElement) elements.videoElement.pause();
        
        clearCanvas();
        log("Analysis pipeline stopped.", "AI");
    }

    function resetSequence() {
        state.currentStepIndex = 0;
        state.validationState = 'VALID';
        state.confidence = 96;
        renderProcedureSteps();
        updateStatusUI();
        log("Procedure sequence reset to Step 1.", "SYS");
        speakVoice("Procedure sequence reset to Step 1.");
    }

    // Canvas Bounding Box Drawer
    function startCanvasDrawing() {
        const canvas = elements.aiCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        let phase = 0;

        function animate() {
            if (!state.isAnalyzing) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            if (!state.isPaused) {
                phase += 0.05;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw bounding box for human / hand interaction
                const boxWidth = canvas.width * 0.35;
                const boxHeight = canvas.height * 0.55;
                const x = (canvas.width - boxWidth) / 2 + Math.sin(phase) * 15;
                const y = (canvas.height - boxHeight) / 2 + Math.cos(phase * 0.7) * 10;

                // Glowing neon cyan box
                ctx.strokeStyle = '#00daf3';
                ctx.lineWidth = 2;
                ctx.setLineDash([8, 4]);
                ctx.strokeRect(x, y, boxWidth, boxHeight);
                ctx.setLineDash([]);

                // Corner accents
                const cornerSize = 12;
                ctx.strokeStyle = '#00daf3';
                ctx.lineWidth = 3;

                // Top-Left
                ctx.beginPath(); ctx.moveTo(x, y + cornerSize); ctx.lineTo(x, y); ctx.lineTo(x + cornerSize, y); ctx.stroke();
                // Top-Right
                ctx.beginPath(); ctx.moveTo(x + boxWidth - cornerSize, y); ctx.lineTo(x + boxWidth, y); ctx.lineTo(x + boxWidth, y + cornerSize); ctx.stroke();
                // Bottom-Left
                ctx.beginPath(); ctx.moveTo(x, y + boxHeight - cornerSize); ctx.lineTo(x, y + boxHeight); ctx.lineTo(x + cornerSize, y + boxHeight); ctx.stroke();
                // Bottom-Right
                ctx.beginPath(); ctx.moveTo(x + boxWidth - cornerSize, y + boxHeight); ctx.lineTo(x + boxWidth, y + boxHeight); ctx.lineTo(x + boxWidth, y + boxHeight - cornerSize); ctx.stroke();

                // Target label tag
                ctx.fillStyle = 'rgba(0, 218, 243, 0.85)';
                ctx.fillRect(x, y - 22, 130, 20);
                ctx.fillStyle = '#001f24';
                ctx.font = 'bold 11px "JetBrains Mono", monospace';
                ctx.fillText(`SUBJECT_01 | ${state.confidence}%`, x + 6, y - 8);
            }
            requestAnimationFrame(animate);
        }
        animate();
    }

    function clearCanvas() {
        const canvas = elements.aiCanvas;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }

    function startSimulationLoop() {
        clearInterval(state.analysisInterval);
        state.analysisInterval = setInterval(() => {
            if (!state.isAnalyzing || state.isPaused) return;

            // Random slight variance in confidence
            state.confidence = Math.min(99, Math.max(88, Math.floor(92 + Math.random() * 7)));
            
            const currentStep = experimentSteps[state.currentStepIndex];
            if (elements.aiActionText) {
                elements.aiActionText.textContent = `Action: ${currentStep.expectedAction}`;
            }

            updateStatusUI();
        }, 3000);
    }

    // Attach Main Controls
    if (elements.btnStartAnalysis) elements.btnStartAnalysis.addEventListener('click', startAnalysis);
    if (elements.btnPauseAnalysis) elements.btnPauseAnalysis.addEventListener('click', pauseAnalysis);
    if (elements.btnStopAnalysis) elements.btnStopAnalysis.addEventListener('click', stopAnalysis);
    if (elements.btnResetSeq) elements.btnResetSeq.addEventListener('click', resetSequence);

    // Initial render
    renderProcedureSteps();
    updateStatusUI();
    setSourceStandby();
});
