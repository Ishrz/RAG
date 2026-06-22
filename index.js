import fs from "node:fs/promises"
import {PDFParse} from "pdf-parse"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
const pdfBuffer = await fs.readFile("./story.pdf")
import * as cheerio from "cheerio";

const pdfUnit8Array = new Uint8Array(pdfBuffer)

const parser = new PDFParse(pdfUnit8Array)
// console.log(parser)

let doc = await parser.getText()
// console.log(doc.text)
// doc =doc.text

const extractedText = typeof doc === "object" && doc !== null ? doc.text : doc


const splitter = new RecursiveCharacterTextSplitter({
    chunkSize:900,
    chunkOverlap:0
})

const splitDoc = await splitter.splitText(extractedText)

// console.log(splitDoc.length)



