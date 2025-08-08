# from openai import OpenAI
# from config import OPEN_API_KEY
# client = OpenAI(api_key=OPEN_API_KEY)

# response = client.responses.create(
#     model="gpt-5-mini",
#     max_output_tokens=1000,
#     input=
#     """
#     Given the following code, generate a clear and specific question that this code would be an appropriate answer to. The question should be practical and educational, focusing on what the code does, how it works, or what problem it solves.

# Code:
# ```
# n=31-lt(t),r=1<<n;e[n]=-1,t&=~r}}function us(e){if(0!==(6&Pu))throw Error(o(327));ks();var t=ft(e,0);if(0===(1&t))return rs(e,Xe()),null;var n=vs(e,t);if(0!==e.tag&&2===n){var r=ht(e);0!==r&&(t=r,n=os(e,r))}if(1===n)throw n=Lu,fs(e,0),is(e,t),rs(e,Xe()),n;if(6===n)throw Error(o(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,xs(e,Iu,Hu),rs(e,Xe()),null}function ss(e,t){var n=Pu;Pu|=1;try{return e(t)}finally{0===(Pu=n)&&(Bu=Xe()+500,Ua&&Ba())}}function cs(e){null!==$u&&0===$u.tag&&0===(6&Pu)&&ks();var t=Pu;Pu|=1;var n=Nu.transition,r=bt;try{if(Nu.transition=null,bt=1,e)return e()}finally{bt=r,Nu.transition=n,0===(6&(Pu=t))&&Ba()}}function ds(){Ou=zu.current,Ea(zu)}function fs(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(-1!==n&&(e.timeoutHandle=-1,aa(n)),null!==Tu)for(n=Tu.return;null!==n;){var r=n;switch(to(r),r.tag){case 1:null!==(r=r.type.childContextTypes)&&void 0!==r&&za();break;case
# ```

# Generate only the question in natural language, without any code blocks, formatting, or additional explanation. The question should be direct and suitable for a programming Q&A dataset.

#     """
# )

# print(response.output_text)





from openai import OpenAI
from config import OPEN_API_KEY
client = OpenAI(api_key=OPEN_API_KEY)

response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content":
    """
    Given the following code, generate a clear and specific question that this code would be an appropriate answer to. The question should be practical and educational, focusing on what the code does, how it works, or what problem it solves.

Code:
```
n=31-lt(t),r=1<<n;e[n]=-1,t&=~r}}function us(e){if(0!==(6&Pu))throw Error(o(327));ks();var t=ft(e,0);if(0===(1&t))return rs(e,Xe()),null;var n=vs(e,t);if(0!==e.tag&&2===n){var r=ht(e);0!==r&&(t=r,n=os(e,r))}if(1===n)throw n=Lu,fs(e,0),is(e,t),rs(e,Xe()),n;if(6===n)throw Error(o(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,xs(e,Iu,Hu),rs(e,Xe()),null}function ss(e,t){var n=Pu;Pu|=1;try{return e(t)}finally{0===(Pu=n)&&(Bu=Xe()+500,Ua&&Ba())}}function cs(e){null!==$u&&0===$u.tag&&0===(6&Pu)&&ks();var t=Pu;Pu|=1;var n=Nu.transition,r=bt;try{if(Nu.transition=null,bt=1,e)return e()}finally{bt=r,Nu.transition=n,0===(6&(Pu=t))&&Ba()}}function ds(){Ou=zu.current,Ea(zu)}function fs(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(-1!==n&&(e.timeoutHandle=-1,aa(n)),null!==Tu)for(n=Tu.return;null!==n;){var r=n;switch(to(r),r.tag){case 1:null!==(r=r.type.childContextTypes)&&void 0!==r&&za();break;case
```

Generate only the question in natural language, without any code blocks, formatting, or additional explanation. Don't make observations.

    """}],
    max_completion_tokens=1000,
    temperature=1,
)

print(response.choices[0].message.content)