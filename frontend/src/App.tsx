import './App.css'
import { useRef, useState } from 'react'
import { Header } from './components/layout/Header'
import { MainLayout } from './components/layout/MainLayout'
import { VideoPanel } from './components/video/VideoPanel'
import { VerdictPanel } from './components/verdict/VerdictPanel'
import { BottomTabs } from './components/layout/BottomTabs'
import { ErrorToastProvider } from './context/ErrorToastContext'
import type { AnalysisResult, ClipMeta, XaiMode } from './types/analysis'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [currentClip, setCurrentClip] = useState<ClipMeta | null>(null)
  const [xaiMode, setXaiMode] = useState<XaiMode>('lrp')
  const [isScanning, setIsScanning] = useState(false)

  return (
    <ErrorToastProvider>
      <Header />
      <MainLayout
        left={
          <VideoPanel
            videoRef={videoRef}
            onResult={setResult}
            onClipChange={setCurrentClip}
            onScanningChange={setIsScanning}
            xaiMode={xaiMode}
            onXaiModeChange={setXaiMode}
          />
        }
        right={
          <VerdictPanel
            result={result}
            clip={currentClip}
            isScanning={isScanning}
            xaiMode={xaiMode}
            onXaiModeChange={setXaiMode}
          />
        }
        bottom={
          <BottomTabs result={result} clip={currentClip} videoRef={videoRef} />
        }
      />
    </ErrorToastProvider>
  )
}

export default App
