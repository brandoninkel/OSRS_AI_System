import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5001'

// Popular OSRS items for autocomplete
const POPULAR_ITEMS = [
  'Abyssal whip', 'Dragon scimitar', 'Bandos chestplate', 'Armadyl chestplate',
  'Twisted bow', 'Scythe of vitur', 'Dragon claws', 'Elysian spirit shield',
  'Ancestral robe top', 'Ancestral robe bottom', 'Kodai wand', 'Dragon warhammer',
  'Toxic blowpipe', 'Dragon hunter lance', 'Ghrazi rapier', 'Inquisitor\'s hauberk',
  'Avernic defender', 'Primordial boots', 'Pegasian boots', 'Eternal boots',
  'Amulet of fury', 'Amulet of torture', 'Necklace of anguish', 'Occult necklace',
  'Berserker ring (i)', 'Archers ring (i)', 'Seers ring (i)', 'Ring of suffering (i)',
  'Dragon boots', 'Ranger boots', 'Infinity boots', 'Mystic boots',
  'Black d\'hide body', 'Red d\'hide body', 'Rune platebody', 'Rune chainbody',
  'Dragon platelegs', 'Dragon plateskirt', 'Barrows gloves', 'Dragon defender',
  'Saradomin godsword', 'Armadyl godsword', 'Zamorak godsword', 'Bandos godsword',
  'Abyssal dagger', 'Dragon dagger', 'Rune crossbow', 'Dragon crossbow',
  'Magic shortbow', 'Toxic staff of the dead', 'Staff of light', 'Trident of the seas',
  'Trident of the swamp', 'Iban\'s staff', 'Ancient staff', 'Master wand',
  'Rune arrow', 'Dragon arrow', 'Amethyst arrow', 'Diamond bolts (e)',
  'Onyx bolts (e)', 'Dragon bolts (e)', 'Rune dart', 'Dragon dart',
  'Shark', 'Manta ray', 'Anglerfish', 'Dark crab', 'Saradomin brew(4)',
  'Super restore(4)', 'Prayer potion(4)', 'Super combat potion(4)', 'Ranging potion(4)',
  'Magic potion(4)', 'Stamina potion(4)', 'Anti-venom+(4)', 'Sanfew serum(4)',
  'Dragon bones', 'Superior dragon bones', 'Ensouled dragon head', 'Big bones',
  'Ranarr seed', 'Snapdragon seed', 'Torstol seed', 'Magic logs', 'Yew logs',
  'Runite ore', 'Adamantite ore', 'Coal', 'Gold ore', 'Rune bar', 'Adamant bar'
]

export default function EconomicDashboard() {
  // Single item lookup state
  const [itemName, setItemName] = useState('')
  const [timeRange, setTimeRange] = useState(24)
  const [loading, setLoading] = useState(false)
  const [priceData, setPriceData] = useState(null)
  const [error, setError] = useState(null)
  
  // Comparison state
  const [compareInput, setCompareInput] = useState('')
  const [selectedItems, setSelectedItems] = useState([])
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareData, setCompareData] = useState(null)
  
  // Autocomplete state
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  const [trackedItems, setTrackedItems] = useState([])
  const inputRef = useRef(null)

  // Fetch tracked items on mount
  useEffect(() => {
    fetchTrackedItems()
  }, [])

  async function fetchTrackedItems() {
    try {
      const response = await fetch(`${API_BASE}/economic/tracked_items?limit=100`)
      const data = await response.json()
      if (data.success) {
        setTrackedItems(data.items)
      }
    } catch (err) {
      console.error('Failed to fetch tracked items:', err)
    }
  }

  // Filter suggestions based on input - now uses wiki search API
  async function updateSuggestions(input) {
    if (!input.trim()) {
      setSuggestions([])
      setShowSuggestions(false)
      return
    }

    try {
      // Search GE items via API (uses actual GE database)
      const response = await fetch(`${API_BASE}/economic/search_items?q=${encodeURIComponent(input)}&limit=10`)
      const data = await response.json()

      if (data.success && data.results) {
        setSuggestions(data.results)
        setShowSuggestions(data.results.length > 0)
      } else {
        // Fallback to local popular items if API fails
        const searchTerm = input.toLowerCase()
        const allItems = [...new Set([...trackedItems, ...POPULAR_ITEMS])]
        const filtered = allItems
          .filter(item => item.toLowerCase().includes(searchTerm))
          .slice(0, 10)
        setSuggestions(filtered)
        setShowSuggestions(filtered.length > 0)
      }
    } catch (err) {
      console.error('Failed to search wiki pages:', err)
      // Fallback to local popular items
      const searchTerm = input.toLowerCase()
      const allItems = [...new Set([...trackedItems, ...POPULAR_ITEMS])]
      const filtered = allItems
        .filter(item => item.toLowerCase().includes(searchTerm))
        .slice(0, 10)
      setSuggestions(filtered)
      setShowSuggestions(filtered.length > 0)
    }
  }

  // Single item price lookup
  async function fetchPriceHistory() {
    if (!itemName.trim()) return
    
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(
        `${API_BASE}/economic/price-history?item=${encodeURIComponent(itemName)}&hours=${timeRange}`
      )
      const data = await response.json()
      
      if (data.success) {
        setPriceData(data)
      } else {
        setError(data.error || 'Failed to fetch price data')
      }
    } catch (err) {
      setError('Network error: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  // Add item to comparison
  function addItemToComparison(item) {
    if (!selectedItems.includes(item)) {
      setSelectedItems([...selectedItems, item])
    }
    setCompareInput('')
    setSuggestions([])
    setShowSuggestions(false)
  }

  // Remove item from comparison
  function removeItemFromComparison(item) {
    setSelectedItems(selectedItems.filter(i => i !== item))
  }

  // Compare multiple items
  async function compareMultipleItems() {
    if (selectedItems.length === 0) return
    
    setCompareLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_BASE}/economic/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: selectedItems, hours: timeRange })
      })
      const data = await response.json()
      
      if (data.success) {
        setCompareData(data)
      } else {
        setError(data.error || 'Failed to compare items')
      }
    } catch (err) {
      setError('Network error: ' + err.message)
    } finally {
      setCompareLoading(false)
    }
  }

  function formatGP(amount) {
    if (!amount) return 'N/A'
    if (amount >= 1000000) return `${(amount / 1000000).toFixed(2)}M`
    if (amount >= 1000) return `${(amount / 1000).toFixed(1)}K`
    return amount.toString()
  }

  function getTrendIcon(trend) {
    if (trend === 'rising') return '📈'
    if (trend === 'falling') return '📉'
    return '➡️'
  }

  function getTrendColor(trend) {
    if (trend === 'rising') return 'text-green-400'
    if (trend === 'falling') return 'text-red-400'
    return 'text-yellow-400'
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold text-foreground">Economic Dashboard</h2>
        <p className="text-muted-foreground mt-1">
          Track item prices, analyze trends, and compare profitability
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <Card className="border-red-500/50 bg-red-500/10">
          <CardContent className="pt-6">
            <p className="text-red-400">❌ {error}</p>
          </CardContent>
        </Card>
      )}

      {/* Price Lookup Section */}
      <Card>
        <CardHeader>
          <CardTitle>💰 Price Lookup</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-3">
            <Input
              placeholder="Item name (e.g., Abyssal whip)"
              value={itemName}
              onChange={e => setItemName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && fetchPriceHistory()}
              className="flex-1"
            />
            <select
              value={timeRange}
              onChange={e => setTimeRange(Number(e.target.value))}
              className="px-3 py-2 rounded-md border bg-background text-foreground"
            >
              <option value={24}>24 hours</option>
              <option value={168}>7 days</option>
              <option value={336}>14 days</option>
              <option value={720}>30 days</option>
              <option value={2160}>90 days</option>
              <option value={4320}>180 days</option>
              <option value={8760}>1 year</option>
            </select>
            <Button onClick={fetchPriceHistory} disabled={loading}>
              {loading ? 'Loading...' : 'Search'}
            </Button>
          </div>

          {/* Price Data Display */}
          {priceData && priceData.trend && (
            <div className="space-y-4 mt-6">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold">{priceData.item}</h3>
                <Badge variant="outline" className={getTrendColor(priceData.trend.trend)}>
                  {getTrendIcon(priceData.trend.trend)} {priceData.trend.trend.toUpperCase()}
                </Badge>
              </div>

              <Separator />

              {priceData.trend.trend === 'insufficient_data' ? (
                <p className="text-muted-foreground">
                  ⚠️ {priceData.trend.message}
                </p>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Current Price</p>
                    <p className="text-2xl font-bold text-foreground">
                      {formatGP(priceData.trend.last_price)} GP
                    </p>
                  </div>

                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Change</p>
                    <p className={`text-2xl font-bold ${priceData.trend.price_change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {priceData.trend.price_change >= 0 ? '+' : ''}{formatGP(priceData.trend.price_change)} GP
                    </p>
                    <p className="text-xs text-muted-foreground">
                      ({priceData.trend.percent_change >= 0 ? '+' : ''}{priceData.trend.percent_change}%)
                    </p>
                  </div>

                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Average</p>
                    <p className="text-2xl font-bold text-foreground">
                      {formatGP(priceData.trend.avg_price)} GP
                    </p>
                  </div>

                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">Volatility</p>
                    <p className="text-2xl font-bold text-yellow-400">
                      ±{formatGP(priceData.trend.volatility)} GP
                    </p>
                  </div>
                </div>
              )}

              {priceData.trend.trend !== 'insufficient_data' && (
                <>
                  <Separator />
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Highest Price</p>
                      <p className="text-lg font-semibold text-green-400">
                        {formatGP(priceData.trend.highest_price)} GP
                      </p>
                    </div>

                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Lowest Price</p>
                      <p className="text-lg font-semibold text-red-400">
                        {formatGP(priceData.trend.lowest_price)} GP
                      </p>
                    </div>
                  </div>

                  <div className="text-xs text-muted-foreground">
                    📊 Based on {priceData.trend.data_points} data points over {priceData.trend.time_range_hours} hours
                  </div>
                </>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Item Comparison Section */}
      <Card>
        <CardHeader>
          <CardTitle>📊 Compare Items</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="relative">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Input
                  ref={inputRef}
                  placeholder="Search for items to compare..."
                  value={compareInput}
                  onChange={e => {
                    setCompareInput(e.target.value)
                    updateSuggestions(e.target.value)
                  }}
                  onFocus={() => updateSuggestions(compareInput)}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                  className="w-full"
                />
                
                {/* Autocomplete Dropdown */}
                {showSuggestions && suggestions.length > 0 && (
                  <div className="absolute z-10 w-full mt-1 bg-background border rounded-md shadow-lg max-h-60 overflow-y-auto">
                    {suggestions.map((item, idx) => (
                      <div
                        key={idx}
                        className="px-4 py-2 hover:bg-secondary cursor-pointer text-sm"
                        onClick={() => addItemToComparison(item)}
                      >
                        {item}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Selected Items Cards */}
          {selectedItems.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-4">
              {selectedItems.map((item, idx) => (
                <Badge
                  key={idx}
                  variant="secondary"
                  className="px-3 py-2 cursor-pointer hover:bg-red-500/20 transition-colors"
                  onClick={() => removeItemFromComparison(item)}
                >
                  {item} ✕
                </Badge>
              ))}
            </div>
          )}

          {/* Compare Button */}
          {selectedItems.length > 0 && (
            <Button 
              onClick={compareMultipleItems} 
              disabled={compareLoading}
              className="w-full"
            >
              {compareLoading ? 'Loading...' : `Compare ${selectedItems.length} Item${selectedItems.length > 1 ? 's' : ''}`}
            </Button>
          )}

          {/* Comparison Results */}
          {compareData && compareData.trends && (
            <div className="space-y-3 mt-6">
              {compareData.trends.map((trend, idx) => (
                <Card key={idx} className="bg-secondary/20">
                  <CardContent className="pt-6">
                    {trend.trend === 'insufficient_data' ? (
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">{trend.item_name}</span>
                        <Badge variant="outline" className="text-muted-foreground">
                          ⚠️ No Data
                        </Badge>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-lg">{trend.item_name}</span>
                          <Badge variant="outline" className={getTrendColor(trend.trend)}>
                            {getTrendIcon(trend.trend)} {trend.trend.toUpperCase()}
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <p className="text-muted-foreground">Current</p>
                            <p className="font-semibold">{formatGP(trend.last_price)} GP</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Change</p>
                            <p className={`font-semibold ${trend.price_change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {trend.price_change >= 0 ? '+' : ''}{formatGP(trend.price_change)} GP
                              <span className="text-xs ml-1">({trend.percent_change}%)</span>
                            </p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Average</p>
                            <p className="font-semibold">{formatGP(trend.avg_price)} GP</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Info Card */}
      <Card className="bg-blue-500/10 border-blue-500/50">
        <CardContent className="pt-6">
          <p className="text-sm text-blue-300">
            💡 <strong>Tip:</strong> Price data is automatically recorded when you ask the AI about item prices. 
            The more you use the system, the more historical data will be available for trend analysis.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
