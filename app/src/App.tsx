import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu,
  Zap,
  Terminal,
  FileCode,
  Copy,
  Check,
  Download,
  AlertTriangle,
  RotateCcw,
  Compass,
  Radio,
  Sliders,
  FolderTree,
  Wifi,
  WifiOff,
  Folder,
  File,
  Bug,
  Server,
  Camera,
  Layers2,
  Eye,
  SlidersHorizontal
} from 'lucide-react';

interface ProjectNode {
  name: string;
  type: 'file' | 'dir';
  status?: 'fixed' | 'missing' | 'error' | 'ok';
  description?: string;
  children?: ProjectNode[];
}

const PROJECT_TREE: ProjectNode = {
  name: 'KOSIARKA-Yolo11-main',
  type: 'dir',
  children: [
    {
      name: 'settings.gradle.kts',
      type: 'file',
      status: 'fixed',
      description: 'Zarządzanie repozytoriami Google/Maven i modułem :app'
    },
    {
      name: 'build.gradle.kts',
      type: 'file',
      status: 'fixed',
      description: 'Główny plik konfiguracji pluginów Android i Kotlin'
    },
    {
      name: 'gradle.properties',
      type: 'file',
      status: 'fixed',
      description: 'Wyłączony configuration-cache dla zgodności z Kotlin 1.9 i Gradle 9.3'
    },
    {
      name: 'app',
      type: 'dir',
      children: [
        {
          name: 'build.gradle.kts',
          type: 'file',
          status: 'fixed',
          description: 'NDK 28, CMake 3.22.1, abiFilters arm64-v8a, org.java-websocket'
        },
        {
          name: 'src',
          type: 'dir',
          children: [
            {
              name: 'main',
              type: 'dir',
              children: [
                {
                  name: 'AndroidManifest.xml',
                  type: 'file',
                  status: 'fixed',
                  description: 'Uprawnienia kamery i sieci, flagi fullscreen i orientation'
                },
                {
                  name: 'assets',
                  type: 'dir',
                  children: [
                    { name: 'yolo11n.param', type: 'file', status: 'ok', description: 'Struktura grafu YOLO11' },
                    { name: 'yolo11n.bin', type: 'file', status: 'ok', description: 'Wagi sieci w formacie FP16' }
                  ]
                },
                {
                  name: 'cpp',
                  type: 'dir',
                  children: [
                    {
                      name: 'CMakeLists.txt',
                      type: 'file',
                      status: 'fixed',
                      description: 'Kompilacja biblioteki natywnej z OpenMP i 16KB Page Size'
                    },
                    {
                      name: 'yolo11.h',
                      type: 'file',
                      status: 'fixed',
                      description: 'Silnik z rotacją sprzętową Vulkan, diagnostyką surowego wyniku i brakiem wycieków'
                    },
                    {
                      name: 'yolo11.cpp',
                      type: 'file',
                      status: 'fixed',
                      description: 'Zero-Copy AHardwareBuffer na Mali-G72, raportowanie najlepszego kandydata'
                    },
                    {
                      name: 'yolo11ncnn_jni.cpp',
                      type: 'file',
                      status: 'fixed',
                      description: 'JNI DetectAHBDirect z obrotem EXIF, GetTopScore i GetTopClass'
                    }
                  ]
                },
                {
                  name: 'java/com/tencent/yolo11ncnn',
                  type: 'dir',
                  children: [
                    {
                      name: 'MainActivity.kt',
                      type: 'file',
                      status: 'fixed',
                      description: 'Czysty JSON bez cudzysłowów, obsługa obrotu kamery i suwak progu w locie'
                    },
                    {
                      name: 'TelemetryServer.kt',
                      type: 'file',
                      status: 'fixed',
                      description: 'Serwer WebSocket nasłuchujący na porcie 8080 (0.0.0.0)'
                    },
                    {
                      name: 'Yolo11Ncnn.kt',
                      type: 'file',
                      status: 'fixed',
                      description: 'Wrapper JNI z metodami GetTopScore i GetTopClass'
                    },
                    {
                      name: 'STM32UartProtocol.kt',
                      type: 'file',
                      status: 'fixed',
                      description: 'Binarny protokół UART (0xAA 0x55) z sumą kontrolną CRC16'
                    }
                  ]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
};

// COCO 80 classes labels
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'live' | 'deadlock' | 'files' | 'map'>('live');
  const [selectedFileKey, setSelectedFileKey] = useState<string>('MainActivity.kt');
  const [copied, setCopied] = useState(false);

  // WebSocket Live Telemetry State
  const [wsUrl, setWsUrl] = useState<string>('ws://192.168.223.127:8080');
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [isSimulated, setIsSimulated] = useState<boolean>(true);
  const [receivedDetections, setReceivedDetections] = useState<any[]>([]);
  const [isEmergencyStop, setIsEmergencyStop] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(15);
  const [topScore, setTopScore] = useState<number>(0.18);
  const [topClass, setTopClass] = useState<number>(0);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [lastPacketTime, setLastPacketTime] = useState<string>('Oczekiwanie...');
  const [pipelineState, setPipelineState] = useState<{
    server: 'idle' | 'online';
    yolo: 'idle' | 'loading' | 'ready' | 'error';
    camera: 'idle' | 'starting' | 'streaming' | 'error';
    details: string;
  }>({
    server: 'idle',
    yolo: 'idle',
    camera: 'idle',
    details: 'Gotowy do połączenia z telefonem.'
  });

  const [logMessages, setLogMessages] = useState<string[]>([
    'Panel diagnostyczny kosiarki uruchomiony.',
    'Domyślny adres telefonu w Wi-Fi: ws://192.168.223.127:8080',
    'Alternatywa przez kabel USB: adb forward tcp:8080 tcp:8080 i ws://localhost:8080',
    'Dodano diagnostykę TopCandidate (surowy najwyższy wynik sieci przed odcięciem).'
  ]);

  const wsRef = useRef<WebSocket | null>(null);

  // Simulated Live stream ticker
  useEffect(() => {
    if (!isSimulated) return;

    const interval = setInterval(() => {
      const simTop = 0.15 + Math.random() * 0.20;
      setTopScore(simTop);
      setTopClass(Math.random() < 0.7 ? 0 : 56);

      if (simTop >= confThreshold) {
        const hasDanger = Math.random() < 0.35;
        const simBottomY = hasDanger ? 0.72 + Math.random() * 0.15 : 0.42 + Math.random() * 0.22;
        const zoneId = simBottomY > 0.70 ? 0 : simBottomY >= 0.35 ? 1 : 2;

        setReceivedDetections([
          {
            zone: zoneId,
            class: 0,
            score: simTop,
            x: 0.42 + Math.sin(Date.now() / 1500) * 0.15,
            y: simBottomY - 0.14,
            w: 0.18,
            h: 0.14
          }
        ]);
        setIsEmergencyStop(zoneId === 0);
      } else {
        setReceivedDetections([]);
        setIsEmergencyStop(false);
      }

      setFps(Math.floor(13 + Math.random() * 4));
      setLastPacketTime(new Date().toLocaleTimeString());
      setPipelineState({
        server: 'online',
        yolo: 'ready',
        camera: 'streaming',
        details: 'Tryb symulacji: Mali-G72 MP18 (Zero-Copy AHB)'
      });
    }, 150);

    return () => clearInterval(interval);
  }, [isSimulated, confThreshold]);

  // Handle Real WebSocket Connection
  const connectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    setWsStatus('connecting');
    addLog(`Łączenie z ${wsUrl}...`);

    try {
      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        setWsStatus('connected');
        setIsSimulated(false);
        addLog(`Połączono z serwerem na telefonie (${wsUrl})!`);
        setPipelineState(prev => ({ ...prev, server: 'online', details: 'Połączono z serwerem telemetrii telefonu.' }));
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'status') {
            addLog(`STATUS: [${data.status}] ${data.detail || ''}`);
            if (data.status === 'SERVER_ONLINE') {
              setPipelineState(prev => ({ ...prev, server: 'online', details: data.detail || 'Serwer aktywny' }));
            } else if (data.status === 'INIT_YOLO') {
              setPipelineState(prev => ({ ...prev, yolo: 'loading', details: data.detail || 'Ładowanie YOLO Vulkan...' }));
            } else if (data.status === 'YOLO_READY') {
              setPipelineState(prev => ({ ...prev, yolo: 'ready', details: data.detail || 'YOLO Vulkan gotowe!' }));
            } else if (data.status === 'STARTING_CAMERA') {
              setPipelineState(prev => ({ ...prev, camera: 'starting', details: data.detail || 'Uruchamianie kamery...' }));
            } else if (data.status === 'STREAMING') {
              setPipelineState(prev => ({ ...prev, camera: 'streaming', details: data.detail || 'Strumień aktywny' }));
            }
          } else if (data.type === 'telemetry') {
            setReceivedDetections(data.objects || []);
            setIsEmergencyStop(Boolean(data.eStop));
            if (data.fps) {
              setFps(parseFloat(data.fps));
            }
            if (data.topScore !== undefined) {
              setTopScore(parseFloat(data.topScore));
            }
            if (data.topClass !== undefined) {
              setTopClass(parseInt(data.topClass, 10));
            }
            if (data.conf !== undefined) {
              setConfThreshold(parseFloat(data.conf));
            }
            setLastPacketTime(new Date().toLocaleTimeString());
            setPipelineState(prev => ({ ...prev, camera: 'streaming', details: `Klatka: ${data.count} obiektów, TopCandidate: ${(parseFloat(data.topScore || 0) * 100).toFixed(1)}%` }));
          }
        } catch (e) {
          // ignore invalid json
        }
      };

      socket.onerror = () => {
        addLog(`Błąd połączenia z ${wsUrl}. Sprawdź IP lub wykonaj 'adb forward tcp:8080 tcp:8080'.`);
        setWsStatus('disconnected');
      };

      socket.onclose = () => {
        addLog(`Rozłączono z ${wsUrl}.`);
        setWsStatus('disconnected');
      };

      wsRef.current = socket;
    } catch (err: any) {
      addLog(`Wyjątek połączenia: ${err.message}`);
      setWsStatus('disconnected');
    }
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsStatus('disconnected');
    addLog('Rozłączono ręcznie.');
  };

  const sendWsCommand = (cmd: string, extra: Record<string, any> = {}) => {
    const payload = JSON.stringify({ cmd, ...extra });
    addLog(`Wysyłanie: ${payload}`);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(payload);
    } else {
      if (cmd === 'EMERGENCY_STOP') {
        setIsEmergencyStop(true);
        addLog('[SYMULACJA] E-STOP aktywowany.');
      } else if (cmd === 'RESET_ESTOP') {
        setIsEmergencyStop(false);
        addLog('[SYMULACJA] E-STOP zresetowany.');
      } else if (cmd === 'SET_PARAMS') {
        if (extra.conf !== undefined) setConfThreshold(extra.conf);
        addLog(`[SYMULACJA] Zmieniono próg na ${extra.conf}`);
      }
    }
  };

  const updateThreshold = (newConf: number) => {
    setConfThreshold(newConf);
    sendWsCommand('SET_PARAMS', { conf: newConf, nms: 0.45 });
  };

  const addLog = (msg: string) => {
    setLogMessages((prev) => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 30)]);
  };

  const copyText = (txt: string) => {
    navigator.clipboard.writeText(txt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadFile = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const topClassName = (topClass >= 0 && topClass < COCO_CLASSES.length) ? COCO_CLASSES[topClass] : `class_${topClass}`;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30">
      {/* Top Bar */}
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 text-white shadow-lg shadow-emerald-500/20">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-base tracking-wide text-white">Kosiarka YOLO11 Vision Core</h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono">
                Galaxy S9+ Mali-G72 MP18
              </span>
            </div>
            <p className="text-xs text-slate-400">Zero-Copy AHardwareBuffer &bull; Serwer Telemetrii WebSocket &bull; STM32F405</p>
          </div>
        </div>

        {/* Tab navigation */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab('live')}
            className={`px-3.5 py-1.5 rounded-lg font-medium transition flex items-center space-x-1.5 ${
              activeTab === 'live' ? 'bg-emerald-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Radio className="w-4 h-4" />
            <span>Pulpit Telemetrii (Live)</span>
          </button>

          <button
            onClick={() => setActiveTab('deadlock')}
            className={`px-3.5 py-1.5 rounded-lg font-medium transition flex items-center space-x-1.5 ${
              activeTab === 'deadlock' ? 'bg-emerald-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Bug className="w-4 h-4" />
            <span>Diagnostyka</span>
          </button>

          <button
            onClick={() => setActiveTab('files')}
            className={`px-3.5 py-1.5 rounded-lg font-medium transition flex items-center space-x-1.5 ${
              activeTab === 'files' ? 'bg-emerald-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileCode className="w-4 h-4" />
            <span>Poprawione Pliki</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* TAB 1: LIVE WEBSOCKET TELEMETRY */}
        {activeTab === 'live' && (
          <div className="space-y-6">
            {/* Top WebSocket Connection Bar with quick IP buttons */}
            <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center space-x-3 flex-1 min-w-[280px]">
                <div className={`p-2 rounded-xl ${wsStatus === 'connected' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-800 text-slate-400'}`}>
                  {wsStatus === 'connected' ? <Wifi className="w-5 h-5" /> : <WifiOff className="w-5 h-5" />}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-[11px] font-mono text-slate-400 block">
                      Adres WebSocket Serwera (Galaxy S9+ Port 8080):
                    </label>
                    <div className="flex items-center space-x-1.5 text-[10px] font-mono">
                      <button
                        onClick={() => setWsUrl('ws://192.168.223.127:8080')}
                        className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700"
                      >
                        Wi-Fi (192.168.223.127)
                      </button>
                      <button
                        onClick={() => setWsUrl('ws://localhost:8080')}
                        className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                      >
                        USB ADB (localhost)
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="text"
                      value={wsUrl}
                      onChange={(e) => setWsUrl(e.target.value)}
                      placeholder="ws://192.168.223.127:8080"
                      className="bg-slate-950 border border-slate-700 px-3 py-1.5 rounded-lg text-xs font-mono text-cyan-300 w-full focus:outline-none focus:border-emerald-500"
                    />
                    {wsStatus !== 'connected' ? (
                      <button
                        onClick={connectWebSocket}
                        className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold transition shrink-0 shadow"
                      >
                        Połącz
                      </button>
                    ) : (
                      <button
                        onClick={disconnectWebSocket}
                        className="px-4 py-1.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-bold transition shrink-0 shadow"
                      >
                        Rozłącz
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Status and Mode Switcher */}
              <div className="flex items-center space-x-4 border-l border-slate-800 pl-4">
                <div>
                  <span className="text-[10px] text-slate-400 block uppercase font-mono">Status:</span>
                  <span className={`text-xs font-bold font-mono ${
                    wsStatus === 'connected' ? 'text-emerald-400' : wsStatus === 'connecting' ? 'text-amber-400' : 'text-slate-400'
                  }`}>
                    {wsStatus === 'connected' ? 'LIVE (POŁĄCZONO)' : wsStatus === 'connecting' ? 'ŁĄCZENIE...' : 'ROZŁĄCZONY'}
                  </span>
                </div>

                <div className="flex items-center space-x-2 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
                  <span className="text-xs text-slate-300">Symulacja:</span>
                  <button
                    onClick={() => setIsSimulated(!isSimulated)}
                    className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                      isSimulated ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {isSimulated ? 'WŁ' : 'WYŁ'}
                  </button>
                </div>
              </div>
            </div>

            {/* Live Model Vision Inspector (Raw Candidate Inspection & Confidence Threshold Slider) */}
            <div className="p-4 rounded-2xl bg-gradient-to-r from-slate-900 to-slate-900/90 border border-slate-800 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center space-x-2">
                  <Eye className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-bold text-white uppercase font-mono tracking-wider">
                    Diagnostyka Wizyjna: Co Widzi YOLO11 na Wyjściu?
                  </span>
                </div>
                <div className="flex items-center space-x-3 text-xs font-mono">
                  <span className="text-slate-400">Najwyższy wynik w kadrze (Raw Score):</span>
                  <span className={`px-2 py-0.5 rounded font-bold ${topScore >= confThreshold ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-slate-800 text-amber-300'}`}>
                    {(topScore * 100).toFixed(1)}% ({topClassName})
                  </span>
                </div>
              </div>

              {/* Slider for confidence */}
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center space-x-3 flex-1 min-w-[280px]">
                  <SlidersHorizontal className="w-4 h-4 text-emerald-400 shrink-0" />
                  <div className="flex-1">
                    <div className="flex justify-between text-xs font-mono mb-1">
                      <span className="text-slate-300">Próg Pewności (Confidence Threshold):</span>
                      <span className="text-emerald-400 font-bold">{(confThreshold * 100).toFixed(0)}% ({confThreshold.toFixed(2)})</span>
                    </div>
                    <input
                      type="range"
                      min="0.02"
                      max="0.80"
                      step="0.01"
                      value={confThreshold}
                      onChange={(e) => updateThreshold(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => updateThreshold(0.05)}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] font-mono"
                  >
                    Próg 5% (Czuły)
                  </button>
                  <button
                    onClick={() => updateThreshold(0.15)}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] font-mono"
                  >
                    Próg 15% (Średni)
                  </button>
                  <button
                    onClick={() => updateThreshold(0.25)}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] font-mono"
                  >
                    Próg 25% (Standard)
                  </button>
                </div>
              </div>
            </div>

            {/* Live Field of View & Emergency Stop Controls */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Camera Video View with 3-Zone Overlay */}
              <div className="lg:col-span-8 p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <h3 className="font-bold text-white text-sm flex items-center space-x-2">
                      <Compass className="w-4 h-4 text-emerald-400" />
                      <span>Pole Widzenia &bull; Podział Horyzontalny 3 Stref</span>
                    </h3>
                    <span className="text-xs font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-500/30">
                      {fps} FPS (Mali-G72)
                    </span>
                  </div>

                  <span className={`text-xs px-2.5 py-1 rounded-full font-bold font-mono ${
                    isEmergencyStop ? 'bg-red-600 text-white animate-pulse' : 'bg-emerald-600/30 text-emerald-300 border border-emerald-500/40'
                  }`}>
                    {isEmergencyStop ? 'ALARM: E-STOP AKTYWNY (ZATRZYMANO NÓŻ)' : 'STAN BEZPIECZNY'}
                  </span>
                </div>

                {/* Video Area */}
                <div className="relative aspect-[4/3] rounded-xl overflow-hidden border border-slate-700 bg-slate-950 shadow-inner select-none">
                  <div className="absolute inset-0 bg-gradient-to-b from-sky-950/30 via-emerald-950/20 to-green-950/70" />

                  {/* Horizon Line (35%) */}
                  <div className="absolute top-[35%] left-0 right-0 border-b-2 border-dashed border-emerald-500/50 flex items-center justify-between px-3 text-[10px] font-mono text-emerald-400 bg-emerald-950/40 py-0.5">
                    <span>GRANICA HORYZONTU (y = 0.35) &bull; ODOMETRIA WIZYJNA</span>
                    <span>STREFA 2 &uarr; | STREFA 1 &darr;</span>
                  </div>

                  {/* Critical Danger Line (70%) */}
                  <div className="absolute top-[70%] left-0 right-0 border-b-2 border-dashed border-red-500/80 flex items-center justify-between px-3 text-[10px] font-mono text-red-400 bg-red-950/60 py-0.5">
                    <span>STREFA KRYTYCZNA NOŻA (y = 0.70) &bull; TWARDY E-STOP</span>
                    <span>STREFA 1 &uarr; | STREFA 0 &darr;</span>
                  </div>

                  {/* Render Received Obstacles */}
                  {receivedDetections.map((obj, idx) => {
                    const isDanger = obj.zone === 0;
                    const isReaction = obj.zone === 1;
                    const labelName = (obj.class >= 0 && obj.class < COCO_CLASSES.length) ? COCO_CLASSES[obj.class] : `cls_${obj.class}`;
                    return (
                      <div
                        key={idx}
                        style={{
                          left: `${Math.max(5, Math.min(85, obj.x * 100))}%`,
                          top: `${Math.max(10, Math.min(85, obj.y * 100))}%`,
                          width: `${Math.max(12, obj.w * 100)}%`,
                          height: `${Math.max(10, obj.h * 100)}%`
                        }}
                        className={`absolute rounded border-2 shadow-2xl transition-all duration-75 flex flex-col justify-between p-1 ${
                          isDanger
                            ? 'border-red-500 bg-red-500/40 animate-pulse'
                            : isReaction
                            ? 'border-amber-500 bg-amber-500/20'
                            : 'border-emerald-500 bg-emerald-500/20'
                        }`}
                      >
                        <div className="flex items-center justify-between text-[9px] font-bold text-white bg-slate-950/80 px-1 py-0.5 rounded font-mono">
                          <span>{labelName}</span>
                          <span>{(obj.score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="text-[8px] font-mono text-center text-white/90 bg-slate-950/70 rounded">
                          y: {obj.y.toFixed(2)} | bottom: {(obj.y + obj.h).toFixed(2)}
                        </div>
                      </div>
                    );
                  })}

                  <div className="absolute bottom-2 right-3 text-[11px] font-mono text-slate-400 bg-slate-950/90 px-2 py-1 rounded border border-slate-700">
                    Ostrze kosiarki &bull; STM32F405 (y = 1.00)
                  </div>
                </div>

                {/* Control Action Buttons */}
                <div className="flex items-center space-x-3 pt-2">
                  <button
                    onClick={() => sendWsCommand('EMERGENCY_STOP')}
                    className="flex-1 py-3 bg-red-600 hover:bg-red-500 active:bg-red-700 text-white rounded-xl font-bold text-sm transition shadow-lg shadow-red-600/30 flex items-center justify-center space-x-2"
                  >
                    <AlertTriangle className="w-5 h-5" />
                    <span>WYŚLIJ TWARDY E-STOP (ZATRZYMAJ KOSIARKĘ)</span>
                  </button>

                  <button
                    onClick={() => sendWsCommand('RESET_ESTOP')}
                    className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl font-medium text-sm transition border border-slate-700 flex items-center space-x-2"
                  >
                    <RotateCcw className="w-4 h-4 text-emerald-400" />
                    <span>Zresetuj E-STOP</span>
                  </button>
                </div>
              </div>

              {/* Telemetry Sidebar */}
              <div className="lg:col-span-4 space-y-4">
                <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400">Metryki Wizyjne w Czasie Rzeczywistym</h4>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-500 font-mono block">FPS SILNIKA:</span>
                      <span className="text-xl font-bold font-mono text-emerald-400">{fps} FPS</span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-500 font-mono block">WYKRYTE OBIEKTY:</span>
                      <span className="text-xl font-bold font-mono text-cyan-400">{receivedDetections.length}</span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-500 font-mono block">TOP KANDYDAT:</span>
                      <span className="text-lg font-bold font-mono text-amber-300">{(topScore * 100).toFixed(1)}%</span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-500 font-mono block">OSTATNI PAKIET:</span>
                      <span className="text-xs font-bold font-mono text-slate-300">{lastPacketTime}</span>
                    </div>
                  </div>
                </div>

                {/* Console Log Window */}
                <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                      <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Logi Serwera / Telefonu</span>
                    </h4>
                    <span className="text-[10px] text-slate-500 font-mono">{logMessages.length} wpisów</span>
                  </div>

                  <div className="h-48 overflow-y-auto font-mono text-[11px] bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1 text-slate-300">
                    {logMessages.map((msg, i) => (
                      <div key={i} className="leading-tight break-all">
                        {msg}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: DIAGNOSTYKA */}
        {activeTab === 'deadlock' && (
          <div className="space-y-6">
            <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Bug className="w-5 h-5 text-emerald-400" />
                <span>Instrukcja Połączenia Wi-Fi i Diagnostyki w ADB</span>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <span className="text-xs font-mono font-bold text-emerald-400">1. Bezpośrednie połączenie Wi-Fi (IP: 192.168.223.127)</span>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Serwer WebSocket w telefonie nasłuchuje na wszystkich interfejsach sieciowych. Wpisz w panelu:
                  </p>
                  <pre className="p-2.5 bg-slate-900 rounded font-mono text-xs text-cyan-300">ws://192.168.223.127:8080</pre>
                  <p className="text-[11px] text-slate-400">
                    Telefon i komputer muszą być w tej samej sieci Wi-Fi.
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <span className="text-xs font-mono font-bold text-cyan-400">2. Połączenie przez kabel USB (ADB Port Forward)</span>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Jeśli Wi-Fi ma włączoną izolację klientów (AP Isolation), uruchom na komputerze:
                  </p>
                  <pre className="p-2.5 bg-slate-900 rounded font-mono text-xs text-emerald-300">adb forward tcp:8080 tcp:8080</pre>
                  <p className="text-[11px] text-slate-400">
                    A w panelu połącz się z <code>ws://localhost:8080</code>.
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 mt-4">
                <span className="text-xs font-mono font-bold text-amber-300">3. Podgląd logów weryfikacyjnych i surowego TopCandidate w ADB:</span>
                <pre className="p-2.5 bg-slate-900 rounded font-mono text-xs text-emerald-300 select-all">
                  adb logcat -v time -s Mower_Vision_Core Yolo11_ZeroCopy_MaliG72 Yolo11Ncnn_JNI TelemetryServer
                </pre>
                <p className="text-xs text-slate-300">
                  W logach co sekundę pojawi się wpis z surowym wynikiem sieci przed odcięciem progiem:
                </p>
                <code className="text-[11px] text-slate-400 block font-mono bg-slate-900 p-2 rounded">
                  I/Mower_Vision_Core: Inference: FPS=15.2 | Objects=0 | TopCandidate=0.184 (class=0) | ConfThresh=0.25
                </code>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: POPRAWIONE PLIKI */}
        {activeTab === 'files' && (
          <div className="space-y-6">
            <div className="p-4 bg-emerald-950/30 border border-emerald-500/40 rounded-xl text-xs text-emerald-200">
              Wszystkie pliki zostały zaktualizowane: usunięto problematyczne cudzysłowy w Kotlin (naprawa błędu kompilacji), dodano sprzętową rotację EXIF Vulkan oraz diagnostykę <code>TopCandidate</code>.
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-4 space-y-2">
                {['MainActivity.kt', 'yolo11.cpp', 'yolo11.h', 'yolo11ncnn_jni.cpp', 'Yolo11Ncnn.kt', 'TelemetryServer.kt'].map((key) => (
                  <button
                    key={key}
                    onClick={() => setSelectedFileKey(key)}
                    className={`w-full text-left p-3 rounded-xl border transition flex items-center justify-between ${
                      selectedFileKey === key
                        ? 'bg-slate-900 border-emerald-500 shadow-md'
                        : 'bg-slate-950 border-slate-800 hover:bg-slate-900/60'
                    }`}
                  >
                    <div>
                      <span className="text-xs font-mono font-bold text-slate-200 block">{key}</span>
                      <span className="text-[10px] text-slate-400 font-mono">Zaktualizowany i gotowy</span>
                    </div>
                    <span className="text-[9px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono">
                      CZYSTY KOD
                    </span>
                  </button>
                ))}
              </div>

              <div className="lg:col-span-8 p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-mono font-bold text-white">{selectedFileKey}</span>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => {
                        // Dynamically read from the project or provide content
                        const el = document.getElementById('code-content');
                        if (el) copyText(el.innerText);
                      }}
                      className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-mono flex items-center space-x-1.5"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copied ? 'Skopiowano!' : 'Kopiuj'}</span>
                    </button>
                  </div>
                </div>

                <pre id="code-content" className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-[580px] leading-relaxed select-all">
                  {selectedFileKey === 'MainActivity.kt' && `// Skopiuj z pliku /MainActivity.kt w projekcie`}
                  {selectedFileKey === 'yolo11.cpp' && `// Skopiuj z pliku /yolo11.cpp w projekcie`}
                  {selectedFileKey === 'yolo11.h' && `// Skopiuj z pliku /yolo11.h w projekcie`}
                  {selectedFileKey === 'yolo11ncnn_jni.cpp' && `// Skopiuj z pliku /yolo11ncnn_jni.cpp w projekcie`}
                  {selectedFileKey === 'Yolo11Ncnn.kt' && `// Skopiuj z pliku /Yolo11Ncnn.kt w projekcie`}
                  {selectedFileKey === 'TelemetryServer.kt' && `// Skopiuj z pliku /TelemetryServer.kt w projekcie`}
                </pre>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
