import DataCollectionBanner from './components/DataCollectionBanner'
import FileUpload from './components/FileUpload'
import './App.css'

function App() {
  return (
    <div className="app-shell">
      <DataCollectionBanner />
      <div className="app">
        <FileUpload />
      </div>
    </div>
  )
}

export default App
