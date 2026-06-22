import "dotenv/config"
import fs from "node:fs/promises"
import {PDFParse} from "pdf-parse"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import * as cheerio from "cheerio";
import {MistralAIEmbeddings} from "@langchain/mistralai"
import {Index, Pinecone} from "@pinecone-database/pinecone"


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
            text:chunk,
            embedded
        }
})  )

// console.log(embeddedChunks)


const pc = new Pinecone({
    apiKey:process.env.PINECONE_API_KEY
})

const index = await pc.index({host:"https://story-1eduacu.svc.aped-4627-b74a.pinecone.io"})

const result = await index.upsert({
records: embeddedChunks.map( (embeddings,idx) => ({
    id:`doc-${idx}`,
    values:embeddings.embedded,
    metadata:{
        text:embeddings.text
    }
}))
})

console.log(result)









