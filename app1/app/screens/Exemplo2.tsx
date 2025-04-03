import React, { useState, useEffect } from "react";
import { StyleSheet, Text, View, Button } from "react-native";
import axios from "axios";
import Constants from "expo-constants";
import * as FileSystem from "expo-file-system";
import { Asset } from "expo-asset";

const file = require("../assets/output.wav");

export default function Exemplo2() {
  const [transcription, setTranscription] = useState("");
  const [base64Audio, setBase64Audio] = useState<string | null>(null);

  useEffect(() => {
    async function loadAudio() {
      try {
        const asset = Asset.fromModule(file); // Usando o arquivo carregado com require
        await asset.downloadAsync(); // Baixa o arquivo, se necessário
        const base64 = await FileSystem.readAsStringAsync(asset.localUri, {
          encoding: FileSystem.EncodingType.Base64,
        });
        setBase64Audio(base64);
      } catch (error) {
        console.error("Erro ao carregar o áudio", error);
      }
    }

    loadAudio();
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
            encoding: "LINEAR16",  // ou "MULAW" dependendo do seu arquivo
            sampleRateHertz: 16000, // Verifique a taxa real do seu áudio
            languageCode: "en-US",
          },
          audio: {
            content: base64Audio, // Envia o áudio em base64
          },
        },
        {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );

      console.log("Response", response?.data?.results[0]);
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
