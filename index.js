import "dotenv/config"
import fs from "node:fs/promises"
import {PDFParse} from "pdf-parse"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import * as cheerio from "cheerio";
import {MistralAIEmbeddings} from "@langchain/mistralai"


const pdfBuffer = await fs.readFile("./story.pdf")

const pdfUnit8Array = new Uint8Array(pdfBuffer)

const parser = new PDFParse(pdfUnit8Array)
// console.log(parser)

let doc = await parser.getText()
// console.log(doc.text)
// doc =doc.text

const extractedText = typeof doc === "object" && doc !== null ? doc.text : doc


const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:700,
    chunkOverlap:0
})

const chunks = await splitter.splitText(extractedText)

// console.log(splitDoc)

const embedding = new MistralAIEmbeddings({
    apiKey:process.env.MSITRAL_API_KEY,
    model:"mistral-embed"
})

const embeddedChunks = await Promise.all( chunks.map(async (chunk )=> {
        const embedded = await embedding.embedQuery(chunk)
        return{
            content:chunk,
            embedded
        }
})  )

// console.log(embeddedChunks)







