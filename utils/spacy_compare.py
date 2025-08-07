import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load a spaCy model with vectors (e.g., en_core_web_md or en_core_web_lg)
nlp = spacy.load("en_core_web_md")

def get_spacy_embedding(text):
    doc = nlp(text)
    return doc.vector

# Example comparison
emb1 = get_spacy_embedding("for i in range(10): print(i)")
emb2 = get_spacy_embedding("while i < 10: print(i); i += 1")

embed1 = get_spacy_embedding(
"""
const puppeteer = require('puppeteer');
const { KnownDevices } = require('puppeteer');

// const iPhone = KnownDevices['iPhone 6'];

const url = "https://beta.boomerang.trade/";
const browserWS = "ws://127.0.0.1:9222/devtools/browser/62dc34b0-769d-4208-8061-4b812693563d";


async function sendOrder() {
    try {
        // const browser = await puppeteer.launch({ headless: false });
        const browser = await puppeteer.connect({ browserWSEndpoint: browserWS })
        const page = await browser.newPage();
        // // await page.emulate(iPhone);

        await page.goto(url);

        // await page.waitForFunction(() => {
        //     const div = document.querySelector('your-div-selector');
        //     return div && div.getAttribute('aria-hidden') === 'true';
        // });
        await page.waitForSelector('wcm-modal');

        // const els = await page.$$('.flex.justify-center > div > div > div > div');
        setTimeout(async () => {
            const els = await page.$$('.flex.justify-between.text-white.font-bold > div');
        
            for (const el of els) {
                const text = await el.$eval('p', p => p.innerText)
                console.log('text', text)
            }

            // const selectBuySelector = await page.$$('.rounded-lg.focus:bg-gray-400.bg-gray-300.flex.justify-between.items-center.cursor-pointer')
            // const selectBuy = await selectBuySelector.$(a)
            // await selectBuy.click();
        }, 3000)

    } catch (e) {
        console.log(e.message)
    }
}


sendOrder();

// const els = await page.$$('.flex.justify-center > div > div > div > div');
"""
)

embed2 = get_spacy_embedding(
"""
function DataFetch ({openMarket}) {
  
    // const [dbStock, setDbStock] = useState(null)
    const [symbolList, setSymbolList] = useState(null)
    const [symbolName, setSymbolName] = useState('')
    const [stock, setStock] = useState(null)
    //force updates Components
    const [reducerValue, forceUpdate] = useReducer(x => x+1, 0)
    const [filteredData, setFilteredData] = useState([])
    const [searchTerm, setSearchTerm] = useState('')
    const [searchError, setSearchError] = useState('')

    // fetch all stock symbols a to z
    useEffect(() => {
        const fetchSymbolList = async () => {
            const response = await fetch(`${STOCK_SHAPES}/api/tickers`)
            const json = await response.json()

            if (response.ok) {
                setSymbolList(json)
                console.log(json)
                // object
                    // date: "2023-05-12"
                    // isEnabled: true
                    // name: "ALCOA CORP"
                    // symbol: "AA"
            }
        }
        fetchSymbolList()
    }, [])

    //fetch single stock's information
    useEffect(() => {
        if (symbolName) {
          const fetchStock = async () => {
            try {
                    const response = await axios.get(`${STOCK_SHAPES}/api/tickerquote`, {
                params: {
                  symbolName
                }
              });
              setStock(response.data);
            //   console.log('fetched symbol quote');
            //   console.log(stock)
            //   console.log(response.data)
            } catch (error) {
              console.error(error);
            }
          };
          fetchStock();
        }
      }, [symbolName]);
"""
)

similarity = cosine_similarity([embed1], [embed2])
print("Cosine similarity:", similarity[0][0])
