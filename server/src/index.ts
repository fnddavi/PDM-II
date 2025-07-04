import express, { Request, Response } from "express";
import multer from "multer";
import ffmpeg from "fluent-ffmpeg";
import axios from "axios";
import fs from "fs-extra";
import path from "path";
import dotenv from "dotenv";
import cors from "cors";
dotenv.config();

const upload = multer({ dest: "uploads/" });
const GOOGLE_API_TOKEN = process.env.GOOGLE_API_TOKEN as string;

const PORT = process.env.PORT || 3000;
const app = express();
app.use(express.json());
app.use(cors());
app.listen(PORT, () => {
  console.log(`Rodando na porta ${PORT}...`);
});

/*
Para testar é necessário estar na pasta server/data ou alterar o caminho de @context-test.wav
curl -X POST -F "audio=@context-test.wav" http://localhost:3101/upload
*/
app.post(
  "/upload",
  upload.single("audio"),
  async (req: Request, res: Response): Promise<void> => {
    if (!req.file) {
      res.status(400).json({ error: "Nenhum arquivo enviado!" });
    } else {
      const inputFilePath = req.file.path;
      console.log("inputFilePath:", inputFilePath);
      const outputFilePath = path.join(
        "uploads",
        `${Date.now()}-converted.flac`
      );
      try {
        // Converte para FLAC (formato compatível com Google Speech-to-Text)
        await new Promise<void>((resolve, reject) => {
          ffmpeg(inputFilePath)
            .toFormat("flac")
            .audioFrequency(16000) // Define sample rate para 16kHz
            .on("end", () => resolve()) 
            .on("error", reject)
            .save(outputFilePath);
        });

        // Lê e converte o áudio para Base64
        const audioBuffer = await fs.readFile(outputFilePath);
        const base64Audio = audioBuffer.toString("base64");

        // Chama a API de reconhecimento de fala
        const response = await axios.post(
          "https://speech.googleapis.com/v1/speech:recognize",
          {
            config: {
              encoding: "FLAC",
              sampleRateHertz: 16000,
              languageCode: "pt-BR",
            },
            audio: { content: base64Audio },
          },
          {
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${GOOGLE_API_TOKEN}`,
            },
          }
        );

        // Exclui os arquivos temporários
        await fs.remove(inputFilePath);
        await fs.remove(outputFilePath);

        res.json({ transcript: response.data });
      } catch (e: any) {
        res.json({ error: e.message });
      }
    }
  }
);

//aceita qualquer método HTTP ou URL
app.use((_: Request, res: Response) => {
  res.json({ error: "Requisição desconhecida" });
});


