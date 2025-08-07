from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-unimodal")
model = AutoModel.from_pretrained("microsoft/unixcoder-base-unimodal")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# Example
# emb1 = get_embedding("function sum(a, b) { return a + b; }")
# emb2 = get_embedding("const sum = (a, b) => a + b;")

emb1 = get_embedding(
"""# Myapp

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 19.0.5.

## Development server

To start a local development server, run:

```bash
ng serve
```

Once the server is running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Running unit tests

To execute unit tests with the [Karma](https://karma-runner.github.io) test runner, use the following command:

```bash
ng test
```

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.
#a[NUL]n [NUL] g.... all the way to angulartest
[NUL]
"""
)
emb2 = get_embedding(
"""
```jsx
  <BrowserRouter>
    <Routes>
      <Route path="dashboard">
        <Route path="*" element={<Dashboard />} />
      </Route>
    </Routes>
  </BrowserRouter>
  ```

  Now, a `<Link to=".">` and a `<Link to="..">` inside the `Dashboard` component go to the same place! That is definitely not correct!

  Another common issue arose in Data Routers (and Remix) where any `<Form>` should post to it's own route `action` if you the user doesn't specify a form action:

  ```jsx
  let router = createBrowserRouter({
    path: "/dashboard",
    children: [
      {
        path: "*",
        action: dashboardAction,
        Component() {
          // ❌ This form is broken!  It throws a 405 error when it submits because
          // it tries to submit to /dashboard (without the splat value) and the parent
          // `/dashboard` route doesn't have an action
          return <Form method="post">...</Form>;
        },
      },
    ],
  });
  ```

  This is just a compounded issue from the above because the default location for a `Form` to submit to is itself (`"."`) - and if we ignore the splat portion, that now resolves to the parent route.

  **The Solution**
  If you are leveraging this behavior, it's recommended to enable the future flag, move your splat to it's own route, and leverage `../` for any links to "sibling" pages:

  ```jsx
  <BrowserRouter>
    <Routes>
      <Route path="dashboard">
        <Route index path="*" element={<Dashboard />} />
      </Route>
    </Routes>
  </BrowserRouter>
"""
)

embed1 = get_embedding(
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

embed2 = get_embedding(
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

# similarity = cosine_similarity([embed1], [embed2])[0][0]
similarity = cosine_similarity([embed1], [embed2])[0][0]
print("Similarity:", similarity)