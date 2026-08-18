"""Build eval/golden_set.jsonl and prove every passage is verbatim.

A golden_passage that is not a character-for-character substring of its document
makes context_recall measure nothing: the cosine match is against text that is
not in the corpus, so the metric reports a miss the retriever never had a chance
at. This generator refuses to write the file unless every passage is found.
"""
import json
from pathlib import Path

CORPUS = Path("eval/corpus")
METHOD = "havenmark_method.txt"
PATH_ = "havenmark_pathway.txt"
MEDIA = "havenmark_media_briefing.txt"
CALL = "havenmark_market_call.txt"
CHECK = "transaction_checklist.txt"
PORT = "launch_portfolio.txt"
BLUE = "outreach_blueprint.txt"
SOPH = "on_Sophia_pptx.txt"
ALL_DOCS = (METHOD, PATH_, MEDIA, CALL, CHECK, PORT, BLUE, SOPH)


def q(question, expected, passages, doc, needs=("single_passage",),
      type_=("fact",), unanswerable=False):
    return {
        "question": question,
        "unanswerable": unanswerable,
        "expected": expected,
        "golden_passages": list(passages),
        "golden_doc": doc,
        "needs": list(needs),
        "type": list(type_),
    }


R = []

# ---- havenmark_method.txt ----
R += [
    q("What is the Havenmark Method in Chinese, and what does it mean in English?",
      "The Havenmark Method is 制度化 数据化 人性化 长期化 - Systematised, Data-led, Human, Long-term.",
      ["Havenmark方法8个字: 制度化 数据化 人性化 长期化！",
       "The Havenmark Method in four principles: Systematised, Data-led, Human, Long-term."],
      METHOD, type_=["definition"]),
    q("How does Havenmark classify clients, and where should an agent spend their time?",
      "Clients are graded P1, P2, P3 and Watch. Time goes to P1 first, P2 every two weeks, P3 to automated nurture, and no active time on Watch.",
      ["2.1 把客户分四类: P1、P2、P3、观察。时间优先花在P1，P2每两周跟进一次，",
       "Classify every client into four tiers — P1, P2, P3, and Watch."],
      METHOD, type_=["procedure"]),
    q("What three conditions must be met at once for a client to be graded P1?",
      "Financing pre-checked, decision maker identified, and a move-in or completion date within six months. Missing any one makes them P2.",
      ["2.2 A P1 client meets three conditions at once: financing pre-checked, decision",
       "maker identified, and a move-in or completion date within six months."],
      METHOD),
    q("What annual transaction targets does Havenmark set for full-timers and part-timers?",
      "More than 18 cases per year for full-timers and more than 6 for part-timers, against an efficiency target of 2.5x the industry average per head.",
      ["平均年成交目标: 全职 >18 单，兼职 >6 单。",
       "Annual transaction targets: more than 18 cases per year for full-timers, more"],
      METHOD),
    q("What is the target average commission per case, and what happens if an agent falls below it?",
      "Not less than S$9,800. An agent hitting the case count while below it has their pipeline reviewed rather than praised.",
      ["3.3 单均佣金目标不低于 S$9,800。",
       "Target average commission per case: not less than S$9,800."],
      METHOD),
    q("What ramp applies to new agents instead of the full annual target?",
      "4 cases in the first six months and 9 cases in months seven to twelve.",
      ["3.2 New agents are held to a ramp, not the full target: 4 cases in the first six",
       "months, 9 cases in months seven to twelve."],
      METHOD),
    q("What happens if an agent does not update their Friday data?",
      "They receive no new leads the following week.",
      ["4.4 周五数据不更新的顾问，下周不分配新线索。",
       "An agent who does not update Friday data receives no new leads the following week."],
      METHOD, type_=["procedure"]),
    q("What does Havenmark refuse to do when quoting a price?",
      "It will not quote a price that has not been verified against a caveat lodged in the last ninety days.",
      ["6.1 We do not quote a price we have not verified against a caveat lodged in the",
       "last ninety days."],
      METHOD, type_=["negation"]),
]

# ---- havenmark_pathway.txt ----
R += [
    q("What are the five stages of the Havenmark Pathway and how long does each take?",
      "Ground (months 1-3), Frame (4-9), Roof (10-18), Beacon (19-36) and Keystone (month 37 onward).",
      ["Stage 1 — Ground (地基) — Months 1 to 3",
       "Stage 5 — Keystone (基石) — Month 37 onward"],
      PATH_, type_=["definition"]),
    q("What must a new agent do before conducting a viewing alone?",
      "Pass the Practice Test. A new agent may not conduct a viewing alone before passing it, and must shadow 12 viewings first.",
      ["新人在通过实务测试前不得独立带看。",
       "A new agent may not conduct a viewing alone before passing the Practice Test."],
      PATH_, type_=["procedure"]),
    q("What is the pass mark for the Practice Test and how can a candidate fail despite a good total?",
      "The pass mark is 80%, but scoring below 50% in any single section fails the whole test regardless of total.",
      ["Pass mark is 80%. A candidate scoring below 50% in any single section fails the",
       "任何单项低于50%即为不通过，无论总分多少。"],
      PATH_),
    q("What are the four sections of the Practice Test?",
      "Documentation, Valuation, Financing and Conduct, each weighted equally.",
      ["  Section 1 — Documentation. Which form, at which stage, signed by whom.",
       "  Section 4 — Conduct. What must be disclosed and when."],
      PATH_, type_=["definition"]),
    q("What is the default co-broke split and when must a different split be agreed?",
      "Fifty-fifty by default, and any different split must be agreed in writing before the first viewing. A split renegotiated after an offer is not recognised.",
      ["C.2 默认分成为五五分，除非在带看前以书面确认。",
       "The default split is fifty-fifty unless agreed otherwise in writing BEFORE the"],
      PATH_, type_=["procedure"]),
    q("What four things must a co-broke agreement name to be enforceable internally?",
      "Both agencies, both agents, the property, and the split. Missing any of the four makes it unenforceable internally.",
      ["C.3 A co-broke agreement must name both agencies, both agents, the property, and",
       "the split. An agreement missing any of the four is not enforceable internally."],
      PATH_),
    q("How long are Havenmark case files retained?",
      "Seven years.",
      ["案卷保存七年。", "Case files are retained for seven years."],
      PATH_),
    q("What are the three grades of compliance finding and what happens for the most serious?",
      "Minor, Major and Critical. A Critical finding suspends the case immediately and convenes the Practice Committee within one working day.",
      ["Findings are graded Minor, Major or Critical.",
       "重大合规问题一经发现，案件立即暂停。"],
      PATH_, type_=["definition"]),
    q("Who leads the final segment of Market Call, and why is it given to them?",
      "The most junior person present - a Ground agent teaches a rule back, because teaching a rule is how you find out whether you understand it.",
      ["The final segment is deliberately given to the most junior person present.",
       "最后一节由最新人主讲，这是检验理解的方式。"],
      PATH_, type_=["procedure"]),
    q("When an agent leaves Havenmark, what do they take and what stays with the firm?",
      "They keep their own client relationships; case files remain with the firm because the file carries obligations to the client that outlast employment.",
      ["顾问离开时带走客户关系，案卷留在公司。",
       "An agent leaving Havenmark keeps their own client relationships. Case files"],
      PATH_, type_=["procedure"]),
    q("How is a Keystone agent assessed?",
      "Primarily on team retention, and only secondarily on personal transactions.",
      ["基石顾问的考核以团队留存率为主，个人成交为辅。",
       "A Keystone agent is assessed primarily on team retention, and only secondarily"],
      PATH_),
]

# ---- havenmark_media_briefing.txt ----
R += [
    q("What are Havenmark's three content tiers and how often is each published?",
      "Tier 1 Market Read monthly, Tier 2 Unit Study weekly, Tier 3 Answer Post daily.",
      ["Tier 1 — Market Read. Monthly. One district, one question, backed by caveat data.",
       "Tier 3 — Answer Post. Daily. One real client question, answered in under 200"],
      MEDIA, type_=["definition"]),
    q("What is Havenmark's policy on using AI to write posts?",
      "AI may draft but may not publish. Every AI-assisted post is reviewed by a named agent who is accountable for the claim.",
      ["3.1 AI may draft. AI may not publish.", "AI可以起草，不可以直接发布。"],
      MEDIA, type_=["procedure"]),
    q("Can AI be used to generate market figures at Havenmark?",
      "No. Every number must come from a lodged caveat or Havenmark's own transaction records, with the source stated.",
      ["3.3 AI是不可以用来生成市场数据的。所有数字必须来自成交记录。",
       "AI must never be used to generate market figures."],
      MEDIA, type_=["negation"]),
    q("What is the required response time for a portal enquiry during working hours?",
      "20 minutes during working hours, and 12 hours outside them.",
      ["Portal enquiry — 20 minutes during working hours, 12 hours otherwise.",
       "门户网站询问必须在20分钟内回复。"],
      MEDIA),
    q("Why does Havenmark set a twenty-minute response standard?",
      "An enquiry answered after two hours converts at roughly a third of the rate of one answered inside twenty minutes.",
      ["An enquiry answered after two hours converts at roughly a third of the rate of",
       "one answered inside twenty minutes."],
      MEDIA),
    q("Does Havenmark track follower counts?",
      "No. Follower counts are not tracked because followers are not clients and the number is easy to move without moving anything that matters.",
      ["我们不统计粉丝数量。", "Follower counts are not tracked."],
      MEDIA, type_=["negation"]),
    q("What happens to a post that cites a figure without a source?",
      "It is taken down within one working day of being flagged, and it is treated as a Major compliance finding.",
      ["3.4 A post that cites a figure without a source is taken down within one working",
       "day of being flagged. This is a Major compliance finding, not an editorial note."],
      MEDIA, type_=["procedure"]),
]

# ---- transaction_checklist.txt ----
R += [
    q("What documents are required at the offer stage for a rental case versus a sales case?",
      "Rental needs a Letter of Intent and a good faith deposit receipt of one month's rent; sales needs an Option to Purchase and an option fee receipt of 1% of the purchase price.",
      ["Offer Stage | - Letter of Intent (LOI)", "- Option to Purchase (OTP)"],
      CHECK, type_=["procedure"]),
    q("What is the standard commission for a residential sale where the seller is represented?",
      "2% of the transacted price, payable by the seller on completion.",
      ["Residential sale, seller represented | Seller | 2% of transacted price | Payable on completion"],
      CHECK),
    q("How long is the option period for a residential sale, and how long from acceptance to completion?",
      "21 calendar days for the option period, and 8 to 12 weeks from acceptance to completion.",
      ["Offer to acceptance | 3 working days | 21 calendar days (option period)",
       "Acceptance to completion | 14 to 30 days | 8 to 12 weeks"],
      CHECK),
    q("When must an agent disclose that they represent both sides of a transaction?",
      "In writing, signed by both parties, before the offer is made.",
      ["Agent represents both sides | Written, signed by both | Before offer is made"],
      CHECK, type_=["procedure"]),
    q("Which financing checkpoints block an Option to Purchase from being issued?",
      "In-principle approval sighted and loan-to-value confirmed against property type - both are blocking before the OTP is issued.",
      ["In-principle approval sighted | Before OTP is issued | Buyer agent | Yes",
       "Loan-to-value confirmed against property type | Before OTP is issued | Buyer agent | Yes"],
      CHECK),
    q("Which handover inventory category does not require a photograph?",
      "Keys and access - the count is recorded instead. Every other category requires a photograph.",
      ["Keys and access | Unit keys, letterbox, access cards, remote controls | No, count recorded"],
      CHECK, type_=["negation"]),
]

# ---- launch_portfolio.txt ----
R += [
    q("Which project in the portfolio has the earliest expected TOP, and what is its indicative psf?",
      "Wrenfield Park, expected Q1 2027, at S$1,880 to S$2,040 psf.",
      ["Wrenfield Park | D23 | 99-year leasehold | 612 | Q1 2027 | S$1,880 - S$2,040"],
      PORT),
    q("How many units does Marrow Gardens have and how many of those are dual-key?",
      "480 units in total, of which 32 are dual-key, and dual-key units price about 4% above equivalent floor area.",
      ["Marrow Gardens | D19 | 99-year leasehold | 480 | Q3 2027 | S$2,050 - S$2,240"],
      PORT),
    q("Which stacks at Kestrel Bay actually face the waterfront?",
      "Stacks 04 through 09 only. Stacks 01 to 03 face the service road and are priced at the lower end of the band.",
      ["Kestrel Bay — Waterfront facing for stacks 04 through 09 only."],
      PORT),
    q("Which project has the highest indicative psf, and what should agents expect from it?",
      "Portland Hollow at S$3,120 to S$3,480 psf. It is freehold, city fringe, 128 units, with a longer sales runway - not a volume project.",
      ["Portland Hollow — Highest indicative PSF in the portfolio."],
      PORT),
    q("Does the standard residential sale commission rate apply to new launches?",
      "No. New launch commission is set by the developer and must be confirmed in writing with the project marketing agency before the first viewing.",
      ["Commission structure for new launches is set by the developer and is not covered by the standard residential sale rate."],
      PORT, type_=["negation"]),
]

# ---- outreach_blueprint.txt ----
R += [
    q("Who owns each module in the outreach blueprint and how often does each run?",
      "Module A Team Leader monthly, Module B Frame agents and above weekly, Module C all agents daily, Module D Practice Committee monthly.",
      ["Module A — Team Leader — Monthly", "Module C — All agents — Daily"],
      BLUE, type_=["definition"]),
    q("What is the conversion path measured in the outreach blueprint?",
      "Post to Enquiry to Response within 20 minutes to Viewing to Offer, measured at every arrow.",
      ["Post -> Enquiry -> Response within 20 minutes -> Viewing -> Offer"],
      BLUE, type_=["procedure"]),
    q("What does the outreach blueprint explicitly not cover?",
      "Paid advertising, co-broke listings and client data handling - each is governed elsewhere.",
      ["What This Blueprint Does Not Cover", "Paid advertising — separate budget approval"],
      BLUE, type_=["negation"]),
]

# ---- havenmark_market_call.txt ----
R += [
    q("What were Havenmark's 1H 2026 results versus 1H 2025?",
      "268 cases versus 214, up 25.2%, with average commission per case rising 12.9% from S$9,140 to S$10,320.",
      ["Total transacted cases | 214 | 268 | +25.2%",
       "成交量同比增长25.2%，单均佣金增长12.9%。"],
      CALL),
    q("What are the July 2026 Projects of the Month listed in the Market Call?",
      "Wrenfield Park (D23) as POM 1 and Kestrel Bay (D15) as POM 2.",
      ["  POM 1 — Wrenfield Park (D23)", "  POM 2 — Kestrel Bay (D15)",
       "七月推荐项目为 Wrenfield Park 与 Kestrel Bay。"],
      CALL),
    q("What is the starting psf and preview date for Kestrel Bay?",
      "Preview opens 19 July 2026 with starting psf of S$2,380.",
      ["Kestrel Bay is POM 2. Preview opens 19 July 2026, with starting psf of"],
      CALL),
    q("How many properties are on the Havenmark July 2026 Luxury slide, and which district has the most?",
      "Five properties, and District 09 has the most with two of the five.",
      ["Havenmark July 2026 Luxury slide lists 5 properties:",
       "District 09 has the most listings on the Luxury slide, with two of the five."],
      CALL),
]

# ---- multi-document ----
R += [
    q("Which two documents both mention Kestrel Bay, and do their psf figures agree?",
      "The launch portfolio lists Kestrel Bay at S$2,380 to S$2,520 psf and the July Market Call gives a starting psf of S$2,380. They agree at the lower bound.",
      ["Kestrel Bay | D15 | 99-year leasehold | 336 | Q4 2027 | S$2,380 - S$2,520",
       "Kestrel Bay is POM 2. Preview opens 19 July 2026, with starting psf of"],
      PORT, needs=["multi_doc"]),
    q("Portland Hollow's entry psf and the Bayshore Road land price are both quoted. Which is higher and where does each appear?",
      "Portland Hollow's indicative S$3,120 psf entry (launch portfolio) is higher than Bayshore Road's S$1,388 psf per plot ratio land price (Market Call).",
      ["Portland Hollow | D09 | Freehold | 128 | Q2 2028 | S$3,120 - S$3,480",
       "Bayshore Road parcel closed at S$1,388 psf per plot ratio."],
      CALL, needs=["multi_doc"]),
    q("A case reached three viewings with nothing signed. What does the method say, and what happened in the case clinic?",
      "The method escalates to the team leader in the same week; in the clinic the root cause was financing, because in-principle approval had never been sighted.",
      ["5.1 Any case where the client has signed nothing after three viewings is escalated",
       "Case A — three viewings, no signed document. Escalated to Team Leader in the"],
      METHOD, needs=["multi_doc"]),
    q("A co-broke split was disputed after an offer. What rule applies and how was it decided?",
      "The default fifty-fifty split stands, because a renegotiation must be agreed in writing before the first viewing. The Practice Committee found for the default split.",
      ["The default split is fifty-fifty unless agreed otherwise in writing BEFORE the",
       "Case B — co-broke split disputed after an offer was made. The Practice Committee"],
      PATH_, needs=["multi_doc"]),
    q("The twenty-minute response standard appears in more than one document. What is the rule and its stated justification?",
      "Portal and WhatsApp enquiries must be answered within 20 minutes in working hours, justified because an enquiry answered after two hours converts at roughly a third of the rate.",
      ["Portal enquiry — 20 minutes during working hours, 12 hours otherwise.",
       "portal response standard, and why it exists — an enquiry answered after two"],
      MEDIA, needs=["multi_doc"]),
]

# ---- on_Sophia_pptx.txt, retained unchanged ----
R += [
    q("What is One Sophia's entry price in psf and its rental yield projection for 2029?",
      "One Sophia's right entry price is $2846 psf (up), with a projected high rental of $10 psf in 2029 equalling a 4.2% rental yield.",
      ["Right Entry Price (safe entry $2846psf up)", "High Rental ($10psf in 2029 = 4.2% RY)"],
      SOPH),
    q("How many residential, office, and shop units does One Sophia have?",
      "One Sophia is a 3-in-1 mixed development with 367 residential units, 122 office units, and 127 shops.",
      ["122 office units", "367 resi units", "127 shops", "3 in 1 MIX DEVELOPEMENT"],
      SOPH),
    q("What does the One Sophia briefing give as the rental yield and entry price rationale?",
      "One Sophia offers a right entry price of $2846 psf up, with high rental of $10 psf in 2029 equalling a 4.2% rental yield.",
      ["Right Entry Price (safe entry $2846psf up)", "High Rental ($10psf in 2029 = 4.2% RY)"],
      SOPH),
    q("What is the exact commission rate Havenmark agents charge for a One Sophia commercial sale?",
      "The briefing shows a buyer-side commission of 2.5% for one specific closed deal, but the documents do not state a standard commission rate for commercial sales generally.",
      ["Comm : 2.5%"], SOPH, type_=["negation"]),
]

# ---- genuinely unanswerable ----
R += [
    q("What is Havenmark's position on using external web search to answer client questions?",
      "Not present in the documents. The corpus covers method, pathway, media, transactions, launches and outreach, but says nothing about external web search.",
      [], "", type_=["negation"], unanswerable=True),
    q("What is the stock ticker symbol of Havenmark Property Group?",
      "Not present in the documents. Nothing in the corpus refers to a listing or a ticker symbol.",
      [], "", type_=["negation"], unanswerable=True),
    q("How many employees work at Havenmark currently?",
      "Not stated. The documents give agents at Frame stage or above (47 in 1H 2026) but never a total headcount.",
      [], "", type_=["negation"], unanswerable=True),
]

# ---- validate ----
cache = {}


def text_of(doc):
    if doc not in cache:
        cache[doc] = (CORPUS / doc).read_text(encoding="utf-8", errors="replace")
    return cache[doc]


bad = []
for r in R:
    if r["unanswerable"]:
        continue
    for p in r["golden_passages"]:
        # A multi_doc passage legitimately lives in the partner document, so a
        # hit anywhere in the corpus counts. What must never pass is a passage
        # that exists in NO document.
        if any(p in text_of(d) for d in ALL_DOCS):
            continue
        bad.append((r["question"][:58], p[:72]))

n_ans = sum(1 for r in R if not r["unanswerable"])
print(f"entries: {len(R)}  answerable: {n_ans}  unanswerable: {len(R) - n_ans}")
if bad:
    print(f"\n!! {len(bad)} PASSAGE(S) NOT FOUND VERBATIM:")
    for qq, p in bad:
        print(f"   Q={qq}\n     passage={p!r}")
    raise SystemExit(1)

out = Path("eval/golden_set.jsonl")
with out.open("w", encoding="utf-8") as f:
    for r in R:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nOK - every passage verbatim. wrote {out} ({out.stat().st_size} bytes)")
