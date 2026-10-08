import { useEffect, useState, type ChangeEvent, type DragEvent, type FormEvent } from 'react'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

type Section = 'overview' | 'diagnosis' | 'watering' | 'plants'

type ApiHealth = {
  status: string
  disease_model: boolean
  watering_model: boolean
  classes: string[]
}

type Treatment = {
  status: string
  severity: string
  treatment: string
  care: string
  prevention: string
  urgent: boolean
}

type Diagnosis = {
  prediction: string | null
  confidence: number | null
  top_predictions: { rank?: number; class: string; confidence: number }[]
  health_status: string | null
  image_quality: string
  quality_issues: string[]
  plant_detected: boolean
  plant_species: string | null
  status: string
  message: string
  treatment: Treatment | null
  gradcam: string | null
  gradcam_error?: string
  elapsed_sec: number
}

type Watering = {
  action: string
  action_confidence: number
  hours_until_watering: number
  water_amount_l: number
  soil_moisture_pct: number
  weather: { temperature_c: number; humidity_pct: number; source: string; location?: string }
  note: string
  schedule_hint: string
}

type Plant = {
  id: string
  name: string
  species: string
  growth_stage: string
  pot_size_l: number
  plant_age_days: number
  soil_type: string
  city?: string
  notes?: string
  health_status: string
  last_watered_at?: string | null
  added_at: string
  health_log?: { prediction: string | null; at: string }[]
  watering?: { action: string; hours_until_watering: number; schedule_hint: string }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

function prettyClass(value: string) {
  return value.replace('___', ' · ').replaceAll('_', ' ')
}

function dateLabel(value?: string | null) {
  if (!value) return 'Not recorded'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Not recorded' : date.toLocaleDateString()
}

function App() {
  const [section, setSection] = useState<Section>('overview')
  const [health, setHealth] = useState<ApiHealth | null>(null)
  const [plants, setPlants] = useState<Plant[]>([])
  const [error, setError] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [today] = useState(() => new Date().toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' }))
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null)
  const [diagnosing, setDiagnosing] = useState(false)
  const [selectedPlant, setSelectedPlant] = useState('')
  const [city, setCity] = useState('')
  const [moisture, setMoisture] = useState('')
  const [watering, setWatering] = useState<Watering | null>(null)
  const [wateringBusy, setWateringBusy] = useState(false)
  const [plantBusy, setPlantBusy] = useState(false)

  useEffect(() => {
    if (!file) return
    let cancelled = false
    const reader = new FileReader()
    reader.onload = () => {
      if (!cancelled && typeof reader.result === 'string') setPreview(reader.result)
    }
    reader.onerror = () => {
      if (!cancelled) setError('Could not read this image file.')
    }
    reader.readAsDataURL(file)
    return () => {
      cancelled = true
      reader.abort()
    }
  }, [file])

  async function refreshData() {
    try {
      const [apiHealth, trackedPlants] = await Promise.all([
        api<ApiHealth>('/api/health'),
        api<Plant[]>('/api/plants'),
      ])
      setHealth(apiHealth)
      setPlants(trackedPlants)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not connect to the farming API.')
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshData(), 0)
    return () => window.clearTimeout(timer)
  }, [])

  function chooseFile(nextFile?: File) {
    if (!nextFile) return
    if (!nextFile.type.startsWith('image/')) {
      setError('Choose an image file to diagnose.')
      return
    }
    setError('')
    setDiagnosis(null)
    setPreview('')
    setFile(nextFile)
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0])
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault()
    chooseFile(event.dataTransfer.files[0])
  }

  async function runDiagnosis() {
    if (!file) {
      setError('Upload a leaf image before starting a diagnosis.')
      return
    }
    setDiagnosing(true)
    setError('')
    try {
      const data = new FormData()
      data.append('file', file)
      data.append('cam', 'true')
      if (selectedPlant) data.append('plant_id', selectedPlant)
      setDiagnosis(await api<Diagnosis>('/api/diagnose', { method: 'POST', body: data }))
      await refreshData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Diagnosis failed.')
    } finally {
      setDiagnosing(false)
    }
  }

  async function getWateringAdvice(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setWateringBusy(true)
    setError('')
    const plant = plants.find((item) => item.id === selectedPlant)
    const body: Record<string, string | number> = {
      species: plant?.species || 'Tomato',
      city: city || plant?.city || '',
    }
    if (selectedPlant) body.plant_id = selectedPlant
    if (moisture !== '') body.soil_moisture_pct = Number(moisture)
    try {
      setWatering(await api<Watering>('/api/watering', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not get a watering recommendation.')
    } finally {
      setWateringBusy(false)
    }
  }

  async function addPlant(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const formElement = event.currentTarget
    const form = new FormData(formElement)
    const body = {
      name: String(form.get('name') || '').trim(),
      species: String(form.get('species') || 'Tomato'),
      growth_stage: String(form.get('growth_stage') || 'vegetative'),
      pot_size_l: Number(form.get('pot_size_l') || 12),
      plant_age_days: Number(form.get('plant_age_days') || 30),
      soil_type: String(form.get('soil_type') || 'loam'),
      city: String(form.get('city') || '').trim() || null,
    }
    if (!body.name) {
      setError('Give your plant a name first.')
      return
    }
    setPlantBusy(true)
    setError('')
    try {
      const created = await api<Plant>('/api/plants', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      setPlants((current) => [created, ...current])
      setSelectedPlant(created.id)
      formElement.reset()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not add the plant.')
    } finally {
      setPlantBusy(false)
    }
  }

  async function markWatered(plant: Plant) {
    setError('')
    try {
      await api(`/api/plants/${encodeURIComponent(plant.id)}/water`, { method: 'POST' })
      await refreshData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not update the watering log.')
    }
  }

  async function removePlant(plant: Plant) {
    setError('')
    try {
      await api(`/api/plants/${encodeURIComponent(plant.id)}`, { method: 'DELETE' })
      setPlants((current) => current.filter((item) => item.id !== plant.id))
      if (selectedPlant === plant.id) setSelectedPlant('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not remove the plant.')
    }
  }

  const sectionTitle: Record<Section, string> = {
    overview: 'Your garden at a glance',
    diagnosis: 'Plant health check',
    watering: 'Watering planner',
    plants: 'Your plants',
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={(event) => { event.preventDefault(); setSection('overview') }}>
          <span className="brand-mark" aria-hidden="true">✳</span>
          <span><strong>verdant</strong><small>GARDEN COMPANION</small></span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {([
            ['overview', 'Overview', '⌂'],
            ['diagnosis', 'Leaf diagnosis', '◉'],
            ['watering', 'Watering planner', '≈'],
            ['plants', 'My plants', '♧'],
          ] as [Section, string, string][]).map(([id, label, icon]) => (
            <button key={id} className={section === id ? 'nav-item active' : 'nav-item'}
              onClick={() => setSection(id)}>
              <span className="nav-icon" aria-hidden="true">{icon}</span>{label}
              {id === 'plants' && plants.length > 0 && <span className="nav-count">{plants.length}</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-tip">
          <span className="tip-icon">✦</span>
          <strong>A little care goes a long way.</strong>
          <p>Check your leaves regularly to catch changes early.</p>
        </div>
        <div className="profile">
          <div className="avatar">G</div>
          <div><strong>Garden keeper</strong><small>Home garden</small></div>
          <span className="profile-dots">···</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Garden <span>/</span> <strong>{sectionTitle[section]}</strong></div>
          <div className="topbar-right">
            <span className={health ? 'connection online' : 'connection'}>
              <i />{health ? 'API connected' : 'API offline'}
            </span>
            <span className="top-date">{today}</span>
          </div>
        </header>

        <div className="page-content">
          {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss">×</button></div>}

          {section === 'overview' && (
            <>
              <section className="welcome-row">
                <div>
                  <div className="eyebrow"><span className="eyebrow-dot" /> YOUR GARDEN, IN GOOD HANDS</div>
                  <h1>Grow well, <em>every day.</em></h1>
                  <p className="intro">A calm little space to understand your plants and help them thrive.</p>
                </div>
                <button className="button primary" onClick={() => setSection('diagnosis')}>＋ Check a leaf</button>
              </section>

              <section className="hero-card">
                <div className="hero-copy">
                  <span className="hero-kicker">A HEALTHIER GARDEN STARTS HERE</span>
                  <h2>Notice a change<br />on a leaf?</h2>
                  <p>Take a photo and get a thoughtful health check, helpful care tips, and a visual explanation.</p>
                  <button className="button light" onClick={() => setSection('diagnosis')}>Start a diagnosis <span>→</span></button>
                </div>
                <div className="hero-art" aria-hidden="true">
                  <div className="sun-disc" />
                  <div className="leaf leaf-one" /><div className="leaf leaf-two" /><div className="leaf leaf-three" />
                  <div className="plant-stem" /><div className="plant-pot"><i /><i /><i /></div>
                  <div className="art-spark spark-one">✳</div><div className="art-spark spark-two">✦</div>
                </div>
              </section>

              <section className="stat-grid">
                <article className="stat-card"><span className="stat-symbol green">♧</span><div><span>IN YOUR GARDEN</span><strong>{plants.length}</strong><small>tracked {plants.length === 1 ? 'plant' : 'plants'}</small></div></article>
                <article className="stat-card"><span className="stat-symbol peach">◉</span><div><span>SUPPORTED CLASSES</span><strong>{health?.classes.length ?? '—'}</strong><small>leaf conditions</small></div></article>
                <article className="stat-card"><span className="stat-symbol yellow">⌁</span><div><span>CARE STATUS</span><strong className="status-word">{plants.some((plant) => plant.health_status === 'ATTENTION') ? 'Check in' : 'Looking good'}</strong><small>{plants.filter((plant) => plant.health_status === 'ATTENTION').length} plants need attention</small></div></article>
              </section>

              <section className="overview-bottom">
                <div className="panel recent-panel">
                  <div className="panel-heading"><div><span className="eyebrow">YOUR COLLECTION</span><h3>Recently added</h3></div><button className="text-button" onClick={() => setSection('plants')}>See all →</button></div>
                  {plants.length === 0
                    ? <EmptyState title="Your garden is ready" text="Add your first plant to keep its care and health notes in one place." action="Add a plant" onAction={() => setSection('plants')} />
                    : plants.slice(0, 3).map((plant) => <PlantRow key={plant.id} plant={plant} onSelect={() => { setSelectedPlant(plant.id); setSection('plants') }} />)}
                </div>
                <div className="panel shortcut-panel">
                  <span className="eyebrow">QUICK ACTION</span><h3>Small steps, happy plants.</h3>
                  <p>Get a watering suggestion based on your plant and local weather.</p>
                  <button className="button outline" onClick={() => setSection('watering')}>Plan watering <span>→</span></button>
                </div>
              </section>
            </>
          )}

          {section === 'diagnosis' && (
            <>
              <PageHeading eyebrow="AI-ASSISTED PLANT CARE" title="A closer look at your leaves." description="Upload a clear photo of one leaf. Your result includes a visual heatmap and practical next steps." />
              <div className="diagnosis-layout">
                <section className="panel upload-panel">
                  <div className="panel-heading"><div><span className="step-number">01</span><h3>Add a leaf photo</h3></div><span className="muted">JPG, PNG, WEBP</span></div>
                  <label className={`drop-zone ${preview ? 'has-image' : ''}`} onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}>
                    {preview ? <img src={preview} alt="Selected leaf preview" /> : <div className="upload-placeholder"><span className="upload-icon">↑</span><strong>Drop your photo here</strong><span>or browse files from your device</span><span className="upload-hint">A bright, close-up leaf photo works best</span></div>}
                    <input type="file" accept="image/*" onChange={handleFileInput} />
                  </label>
                  {file && <div className="file-row"><span className="file-check">✓</span><span>{file.name}</span><button onClick={() => { setFile(null); setDiagnosis(null) }}>Remove</button></div>}
                  <div className="form-row">
                    <label className="field-label" htmlFor="diagnosis-plant">Add this check to a plant <span>Optional</span></label>
                    <select id="diagnosis-plant" value={selectedPlant} onChange={(event) => setSelectedPlant(event.target.value)}>
                      <option value="">No plant selected</option>
                      {plants.map((plant) => <option value={plant.id} key={plant.id}>{plant.name} · {plant.species}</option>)}
                    </select>
                  </div>
                  <button className="button primary full-button" onClick={() => void runDiagnosis()} disabled={!file || diagnosing}>
                    {diagnosing ? <><span className="spinner" /> Looking closely…</> : <>Run leaf check <span>→</span></>}
                  </button>
                  <p className="fine-print">This is an AI screening aid, not a substitute for local agricultural advice.</p>
                </section>

                <section className="panel result-panel">
                  {diagnosis ? <DiagnosisResult result={diagnosis} /> : <div className="waiting-state">
                    <div className="waiting-illustration"><span>✳</span><div className="wait-leaf" /><div className="wait-leaf second" /></div>
                    <span className="eyebrow">YOUR CHECK-IN</span><h3>Your leaf report will appear here.</h3>
                    <p>We’ll identify the closest matching class and share a visual guide for the areas the model focused on.</p>
                    <div className="waiting-points"><span>01 <i>Image quality check</i></span><span>02 <i>Leaf health prediction</i></span><span>03 <i>Care suggestions</i></span></div>
                  </div>}
                </section>
              </div>
            </>
          )}

          {section === 'watering' && (
            <>
              <PageHeading eyebrow="WEATHER-AWARE CARE" title="Water with a little more confidence." description="Get a practical next step using your plant profile, soil moisture, and local weather when available." />
              <div className="watering-layout">
                <form className="panel watering-form" onSubmit={(event) => void getWateringAdvice(event)}>
                  <span className="eyebrow">YOUR PLANT</span><h3>Tell us what you know.</h3>
                  <label className="field-label" htmlFor="watering-plant">Plant</label>
                  <select id="watering-plant" value={selectedPlant} onChange={(event) => setSelectedPlant(event.target.value)}>
                    <option value="">Tomato · general recommendation</option>
                    {plants.map((plant) => <option value={plant.id} key={plant.id}>{plant.name} · {plant.species}</option>)}
                  </select>
                  <label className="field-label" htmlFor="moisture">Soil moisture <span>Optional</span></label>
                  <div className="input-suffix"><input id="moisture" type="number" min="0" max="100" step="1" placeholder="e.g. 45" value={moisture} onChange={(event) => setMoisture(event.target.value)} /><span>%</span></div>
                  <label className="field-label" htmlFor="city">City for live weather <span>Optional</span></label>
                  <input id="city" type="text" placeholder="e.g. Pune" value={city} onChange={(event) => setCity(event.target.value)} />
                  <button className="button primary full-button" disabled={wateringBusy}>{wateringBusy ? <><span className="spinner" /> Checking conditions…</> : <>Get watering advice <span>→</span></>}</button>
                  <p className="fine-print">If live weather is unavailable, the recommendation uses a plant-profile estimate.</p>
                </form>
                <section className="panel watering-result">
                  {watering ? <WateringResult result={watering} /> : <div className="waiting-state compact">
                    <div className="water-drop">≈</div><span className="eyebrow">A GENTLE REMINDER</span><h3>Every plant has its own rhythm.</h3>
                    <p>Choose a plant or share a soil reading to get a tailored watering suggestion.</p>
                  </div>}
                </section>
              </div>
            </>
          )}

          {section === 'plants' && (
            <>
              <PageHeading eyebrow="YOUR GROWING SPACE" title="Get to know your plants." description="Keep a simple record of each plant, its health check-ins, and watering routine." />
              <div className="plants-layout">
                <section className="panel add-plant-panel">
                  <span className="eyebrow">START A PLANT PROFILE</span><h3>Welcome a new plant.</h3>
                  <form onSubmit={(event) => void addPlant(event)}>
                    <label className="field-label" htmlFor="plant-name">Plant name</label><input id="plant-name" name="name" placeholder="e.g. Balcony tomato" required maxLength={60} />
                    <div className="form-grid">
                      <div><label className="field-label" htmlFor="plant-species">Species</label><select id="plant-species" name="species"><option>Tomato</option><option>Pepper</option><option>Potato</option></select></div>
                      <div><label className="field-label" htmlFor="plant-stage">Growth stage</label><select id="plant-stage" name="growth_stage"><option value="vegetative">Vegetative</option><option value="seedling">Seedling</option><option value="flowering">Flowering</option><option value="fruiting">Fruiting</option></select></div>
                      <div><label className="field-label" htmlFor="pot-size">Pot size (L)</label><input id="pot-size" name="pot_size_l" type="number" min="1" max="500" defaultValue="12" /></div>
                      <div><label className="field-label" htmlFor="plant-age">Age (days)</label><input id="plant-age" name="plant_age_days" type="number" min="0" max="2000" defaultValue="30" /></div>
                    </div>
                    <label className="field-label" htmlFor="plant-city">City <span>For local weather</span></label><input id="plant-city" name="city" placeholder="e.g. Pune" />
                    <button className="button primary full-button" disabled={plantBusy}>{plantBusy ? 'Saving plant…' : '＋ Add to my garden'}</button>
                  </form>
                </section>
                <section className="panel plant-list-panel">
                  <div className="panel-heading"><div><span className="eyebrow">YOUR COLLECTION</span><h3>{plants.length} {plants.length === 1 ? 'plant' : 'plants'} tracked</h3></div></div>
                  {plants.length === 0
                    ? <EmptyState title="Your collection is empty" text="Create a profile to start tracking care and health updates." />
                    : <div className="plant-list">{plants.map((plant) => <PlantCard key={plant.id} plant={plant} onWater={() => void markWatered(plant)} onRemove={() => void removePlant(plant)} onWatering={() => { setSelectedPlant(plant.id); setSection('watering') }} />)}</div>}
                </section>
              </div>
            </>
          )}
        </div>
        <footer className="footer"><span>Made for the love of growing.</span><span>AI is a helpful starting point; your local growing conditions matter.</span></footer>
      </main>
    </div>
  )
}

function PageHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <div className="page-heading"><span className="eyebrow"><span className="eyebrow-dot" />{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>
}

function EmptyState({ title, text, action, onAction }: { title: string; text: string; action?: string; onAction?: () => void }) {
  return <div className="empty-state"><span className="empty-sprout">♧</span><strong>{title}</strong><p>{text}</p>{action && onAction && <button className="text-button" onClick={onAction}>{action} →</button>}</div>
}

function PlantRow({ plant, onSelect }: { plant: Plant; onSelect: () => void }) {
  return <button className="plant-row" onClick={onSelect}><span className="plant-avatar">♧</span><span className="plant-row-main"><strong>{plant.name}</strong><small>{plant.species} · {plant.growth_stage}</small></span><span className={`health-pill ${plant.health_status === 'ATTENTION' ? 'attention' : ''}`}>{plant.health_status || 'UNKNOWN'}</span><span className="row-arrow">→</span></button>
}

function PlantCard({ plant, onWater, onRemove, onWatering }: { plant: Plant; onWater: () => void; onRemove: () => void; onWatering: () => void }) {
  return <article className="plant-card">
    <div className="plant-card-top"><span className="plant-avatar large">♧</span><span className={`health-pill ${plant.health_status === 'ATTENTION' ? 'attention' : ''}`}>{plant.health_status || 'UNKNOWN'}</span></div>
    <h4>{plant.name}</h4><p className="plant-species">{plant.species} · {plant.growth_stage}</p>
    <div className="plant-meta"><span>Pot <strong>{plant.pot_size_l} L</strong></span><span>Added <strong>{dateLabel(plant.added_at)}</strong></span></div>
    <div className="care-line"><span className="care-icon">≈</span><span>{plant.watering?.schedule_hint || 'Watering advice available'}</span></div>
    <div className="plant-card-actions"><button className="button small outline" onClick={onWatering}>Watering plan</button><button className="button small light-green" onClick={onWater}>Watered today</button></div>
    <div className="plant-card-bottom"><span>Last watered: {dateLabel(plant.last_watered_at)}</span><button className="remove-button" onClick={onRemove}>Remove</button></div>
  </article>
}

function DiagnosisResult({ result }: { result: Diagnosis }) {
  const pct = result.confidence == null ? null : Math.round(result.confidence * 100)
  return <div className="diagnosis-result">
    <div className="result-title-row"><div><span className="eyebrow">YOUR LEAF CHECK</span><h3>{result.prediction ? prettyClass(result.prediction) : result.status}</h3></div><span className={`result-quality ${result.image_quality === 'GOOD' ? 'good' : 'warn'}`}>{result.image_quality} IMAGE</span></div>
    <p className="result-message">{result.message}</p>
    {result.prediction && <>
      <div className="confidence-block"><div><span>Model confidence</span><strong>{pct}%</strong></div><div className="confidence-track"><i style={{ width: `${pct}%` }} /></div></div>
      <div className="result-facts"><div><span>PLANT</span><strong>{result.plant_species || 'Not identified'}</strong></div><div><span>HEALTH</span><strong>{result.health_status || 'Needs review'}</strong></div><div><span>CHECK TIME</span><strong>{result.elapsed_sec.toFixed(1)} sec</strong></div></div>
      {result.gradcam && <div className="explanation-block"><div className="subheading"><strong>Where the model focused</strong><span>Grad-CAM visual guide</span></div><img src={result.gradcam} alt="Grad-CAM overlay showing image regions relevant to the prediction" /><p>Highlighted areas indicate regions that influenced the prediction; they are not a diagnosis of lesion boundaries.</p></div>}
      {result.gradcam_error && <p className="inline-note">The diagnosis is ready, but the visual explanation could not be generated: {result.gradcam_error}</p>}
      {result.treatment && <div className="treatment-block"><div className="subheading"><strong>Care notes</strong>{result.treatment.urgent && <span className="urgent-label">Prompt attention</span>}</div><div className="care-item"><span className="care-index">01</span><div><strong>Suggested next step</strong><p>{result.treatment.treatment}</p></div></div><div className="care-item"><span className="care-index">02</span><div><strong>Everyday care</strong><p>{result.treatment.care}</p></div></div><div className="care-item"><span className="care-index">03</span><div><strong>Prevention</strong><p>{result.treatment.prevention}</p></div></div><p className="fine-print">Follow local product labels and consult an agricultural professional before treatment.</p></div>}
      <details className="top-predictions"><summary>Other possible matches</summary>{result.top_predictions.map((item) => <div key={item.class}><span>{prettyClass(item.class)}</span><strong>{(item.confidence * 100).toFixed(1)}%</strong></div>)}</details>
    </>}
    {result.quality_issues.length > 0 && <ul className="quality-issues">{result.quality_issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
  </div>
}

function WateringResult({ result }: { result: Watering }) {
  return <div className="watering-advice">
    <div className="watering-result-icon">≈</div><span className="eyebrow">YOUR WATERING CHECK</span>
    <h3>{result.action === 'water_now' ? 'A drink would help today.' : result.action === 'water_later' ? 'Check in again soon.' : 'Your plant can wait.'}</h3>
    <p>{result.note}</p>
    <div className="watering-metrics"><div><span>TIME UNTIL WATER</span><strong>{result.hours_until_watering < 1 ? 'Now' : `${Math.round(result.hours_until_watering)} hrs`}</strong></div><div><span>SUGGESTED AMOUNT</span><strong>{result.water_amount_l} L</strong></div><div><span>SOIL MOISTURE</span><strong>{result.soil_moisture_pct}%</strong></div></div>
    <div className="weather-strip"><span className="weather-sun">☼</span><span><strong>{result.weather.temperature_c}°C</strong> · {result.weather.humidity_pct}% humidity{result.weather.location ? ` · ${result.weather.location}` : ''}</span><small>{result.weather.source}</small></div>
    <p className="fine-print">A model-based estimate, not a soil sensor reading. Check the soil before watering.</p>
  </div>
}

export default App
