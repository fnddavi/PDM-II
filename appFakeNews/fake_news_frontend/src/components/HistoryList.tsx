// src/components/HistoryList.tsx
import React, { useEffect, useState } from 'react';
import { API_BASE_URL } from '../api';
import { Box, Typography, Paper, CircularProgress, Alert, List, ListItem, ListItemText, Divider } from '@mui/material'; // Importar componentes MUI

interface HistoryEntry {
    texto: string;
    classe: string;
    probabilidade: number;
}

const HistoryList: React.FC = () => {
    const [history, setHistory] = useState<HistoryEntry[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/historico`);
                if (!response.ok) {
                    throw new Error('Erro ao buscar o histórico.');
                }
                const data: HistoryEntry[] = await response.json();
                setHistory(data);
            } catch (err: unknown) {
                if (err instanceof Error) {
                    setError(err.message);
                } else {
                    setError('Erro desconhecido ao buscar o histórico.');
                }
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
        // Você pode configurar um polling para atualizar o histórico a cada X segundos se desejar
        // const interval = setInterval(fetchHistory, 5000);
        // return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Box display="flex" alignItems="center" justifyContent="center" sx={{ p: 2 }}>
                <CircularProgress size={24} sx={{ mr: 2 }} />
                <Typography variant="body1">Carregando histórico...</Typography>
            </Box>
        );
    }

    if (error) {
        return (
            <Alert severity="error" sx={{ mt: 2 }}>
                <Typography variant="body1">Erro: {error}</Typography>
            </Alert>
        );
    }

    return (
        <Paper elevation={1} sx={{ p: 3 }}>
            <Typography variant="h5" component="h2" gutterBottom>
                Histórico de Classificações
            </Typography>
            {history.length === 0 ? (
                <Typography variant="body1" sx={{ fontStyle: 'italic', color: '#666' }}>Nenhuma notícia classificada ainda.</Typography>
            ) : (
                <List>
                    {history.map((entry, index) => (
                        <React.Fragment key={index}>
                            <ListItem alignItems="flex-start">
                                <ListItemText
                                    primary={
                                        <Typography variant="h6" component="span" sx={{ display: 'inline' }} color="text.primary">
                                            Notícia {index + 1}:
                                        </Typography>
                                    }
                                    secondary={
                                        <Box component="span" sx={{ display: 'block', mt: 1 }}>
                                            <Typography
                                                sx={{ display: 'block' }}
                                                component="span"
                                                variant="body2"
                                                color="text.secondary"
                                            >
                                                {entry.texto.length > 200 ? entry.texto.substring(0, 200) + '...' : entry.texto}
                                            </Typography>
                                            <Typography variant="body2" color="text.primary" sx={{ mt: 1 }}>
                                                <strong>Classe:</strong>{' '}
                                                <span style={{ color: entry.classe === 'fake' ? 'red' : 'green', fontWeight: 'bold' }}>
                                                    {entry.classe === 'fake' ? 'Fake News' : 'Verdadeira'}
                                                </span>
                                            </Typography>
                                            <Typography variant="body2" color="text.primary">
                                                <strong>Probabilidade:</strong> {(entry.probabilidade * 100).toFixed(2)}%
                                            </Typography>
                                        </Box>
                                    }
                                />
                            </ListItem>
                            {index < history.length - 1 && <Divider component="li" />}
                        </React.Fragment>
                    ))}
                </List>
            )}
        </Paper>
    );
};

export default HistoryList;