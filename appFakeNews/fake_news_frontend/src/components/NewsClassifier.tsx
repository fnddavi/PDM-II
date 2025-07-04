// src/components/NewsClassifier.tsx
import React, { useState } from 'react';
import { API_BASE_URL } from '../api';
import { TextField, Button, Box, Typography, Paper, Alert, CircularProgress } from '@mui/material'; // Importar componentes MUI

interface ClassificationResult {
    classe: string;
    probabilidade: number;
}

const NewsClassifier: React.FC = () => {
    const [newsText, setNewsText] = useState<string>('');
    const [result, setResult] = useState<ClassificationResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        if (!newsText.trim()) {
            setError('O texto da notícia não pode ser vazio.');
            setLoading(false);
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/classificar-noticia`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ texto: newsText }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Erro ao classificar a notícia.');
            }

            const data: ClassificationResult = await response.json();
            setResult(data);
        } catch (err: unknown) {
            if (err instanceof Error) {
                setError(err.message);
            } else {
                setError('Ocorreu um erro desconhecido.');
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <Paper elevation={1} sx={{ p: 3 }}>
            <Typography variant="h5" component="h2" gutterBottom>
                Classificar Notícia
            </Typography>
            <form onSubmit={handleSubmit}>
                <TextField
                    label="Texto da Notícia"
                    multiline
                    rows={6}
                    fullWidth
                    value={newsText}
                    onChange={(e) => setNewsText(e.target.value)}
                    placeholder="Digite o texto da notícia aqui..."
                    variant="outlined"
                    sx={{ mb: 2 }}
                />
                <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    disabled={loading}
                    fullWidth
                    sx={{ height: 48 }} // Altura fixa para o botão
                >
                    {loading ? <CircularProgress size={24} color="inherit" /> : 'Classificar'}
                </Button>
            </form>

            {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
            {result && (
                <Box sx={{ mt: 2, p: 2, bgcolor: result.classe === 'fake' ? '#ffebee' : '#e8f5e9', borderRadius: '4px' }}>
                    <Typography variant="h6">Resultado:</Typography>
                    <Typography variant="body1">
                        <strong>Classe:</strong>{' '}
                        <span style={{ color: result.classe === 'fake' ? 'red' : 'green', fontWeight: 'bold' }}>
                            {result.classe === 'fake' ? 'Fake News' : 'Verdadeira'}
                        </span>
                    </Typography>
                    <Typography variant="body1">
                        <strong>Probabilidade:</strong> {(result.probabilidade * 100).toFixed(2)}%
                    </Typography>
                </Box>
            )}
        </Paper>
    );
};

export default NewsClassifier;
