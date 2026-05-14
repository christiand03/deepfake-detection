import './App.css'
import { useRef, useState } from 'react'
import { Header } from './components/layout/Header'
import { MainLayout } from './components/layout/MainLayout'
import { VideoPanel } from './components/video/VideoPanel'
import { VerdictPanel } from './components/verdict/VerdictPanel'
import { AudioLayers } from './components/audio/AudioLayers'
import { BottomTabs } from './components/layout/BottomTabs'
import { ErrorToastProvider } from './context/ErrorToastContext'
import type { AnalysisResult, ClipMeta } from './types/analysis'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [currentClip, setCurrentClip] = useState<ClipMeta | null>(null)
  const [activeTab, setActiveTab] = useState<'robustness' | 'adversarial' | null>(null)
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
          />
        }
        right={
          <>
            <VerdictPanel result={result} clip={currentClip} isScanning={isScanning} />
            {activeTab === null && (
              <AudioLayers result={result} clip={currentClip} videoRef={videoRef} />
            )}
          </>
        }
        bottom={
          <BottomTabs result={result} activeTab={activeTab} onTabChange={setActiveTab} />
        }
      />
    </ErrorToastProvider>
  )
}

export default App
