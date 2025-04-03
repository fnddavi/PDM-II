import React, { useState, useEffect } from "react";
import { StyleSheet, Text, View, Button } from "react-native";
import axios from "axios";
import Constants from "expo-constants";
import * as FileSystem from "expo-file-system";
import { Asset } from "expo-asset";
import { Buffer } from "buffer"; // Importando a biblioteca correta

const file = require("../assets/context-test.wav");

export default function Exemplo3() {
  const [transcription, setTranscription] = useState("");
  const [base64Audio, setBase64Audio] = useState<string | null>(null);

  useEffect(() => {
    async function loadAndConvertAudio() {
      try {
        const asset = Asset.fromModule(file);
        await asset.downloadAsync();

        // Lê o arquivo WAV como Base64
        const wavBase64 = await FileSystem.readAsStringAsync(asset.localUri, {
          encoding: FileSystem.EncodingType.Base64,
        });

        // Converte Base64 para ArrayBuffer usando Buffer da biblioteca "buffer"
        const wavBuffer = Buffer.from(wavBase64, "base64");

        // Pula o cabeçalho RIFF (44 bytes) e pega os dados PCM
        const pcmBuffer = wavBuffer.subarray(44);

        // Converte os dados PCM para Base64
        const pcmBase64 = Buffer.from(pcmBuffer).toString("base64");

        setBase64Audio(pcmBase64);
      } catch (error) {
        console.error("Erro ao carregar e converter o áudio", error);
      }
    }

    loadAndConvertAudio();
  }, []);

  async function transcribeAudio() {
    if (!base64Audio) {
      console.error("Áudio não carregado.");
      return;
    }

    try {
      const token = await getAccessToken();
      if (!token) {
        console.error("Erro: Token de autenticação inválido.");
        return;
      }

      const response = await axios.post(
        "https://speech.googleapis.com/v1/speech:recognize",
        {
          config: {
            encoding: "LINEAR16",
            sampleRateHertz: 16000, // Ajuste conforme a taxa do áudio original
            languageCode: "en-US",
          },
          audio: {
            content: base64Audio,
          },
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );

      console.log("Response:", response?.data?.results[0]);
      if (response.data && response.data.results) {
        setTranscription(response.data.results[0].alternatives[0].transcript);
      } else {
        console.error("Erro: Nenhuma transcrição retornada pela API.");
      }
    } catch (error: any) {
      console.error("Erro ao transcrever o áudio", error);
    }
  }

  async function getAccessToken() {
    return Constants.expoConfig?.extra?.GCLOUD_ACCESS_TOKEN;
  }

  return (
    <View style={styles.container}>
      <Button title="Testar Áudio Local" onPress={transcribeAudio} />
      <Text style={styles.transcription}>{transcription}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
  transcription: {
    marginTop: 20,
    textAlign: "center",
    padding: 10,
  },
});
