import torch
import chess

piecetonum = {s:i for i,s in enumerate("PNBRQKpnbrqk")}

N = 18
def board_to_tensor(board):
    tens = torch.zeros(18,8,8)
    whitelr = board.has_kingside_castling_rights(chess.WHITE)
    whitegr = board.has_queenside_castling_rights(chess.WHITE)
    blacklr = board.has_kingside_castling_rights(chess.BLACK)
    blackgr = board.has_queenside_castling_rights(chess.BLACK)
    nbhalfmoves= board.halfmove_clock
    turned = board.turn
    for i in range(8):
        for j in range(8):
            p = board.piece_at(8*i+j)
            if p is not None:
                num = piecetonum[p.symbol()]
                tens[num, i , j]=1
    if turned:
        tens[12]=1
    if whitelr:
        tens[13]=1
    if whitegr:
        tens[14]=1
    if blacklr:
        tens[15]=1
    if blackgr:
        tens[16]=1
    tens[17] = nbhalfmoves/100
    return tens #tens[num,rank,file]


DIRECTIONS = [(1,1),(1,0),(1,-1),(0,-1),(0,1),(-1,1),(-1,-1),(-1,0)] 
knight_directions = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]
underpromotedtoi = {chess.KNIGHT:0, chess.BISHOP:1, chess.ROOK: 2}
itounderpromote = {v: k for k, v in underpromotedtoi.items()}
def sign(x):
    return (x > 0) - (x < 0)


def simplemove_to_index(move):
    #just for moves which are not promotion/knights move
    startcase = move.from_square
    dh = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    dv = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    if not (dh == 0 or dv == 0 or abs(dh) == abs(dv)):
        raise ValueError("not simple move")
    distance = max(abs(dv), abs(dh))
    index = (distance-1) + 7*DIRECTIONS.index((sign(dh), sign(dv)))
    return index + 73*startcase

def move_to_index(move):
    startcase = move.from_square
    dh = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    dv = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    if move.promotion is not None:
        if move.promotion ==chess.QUEEN:
            return simplemove_to_index(move)
        else:
            piece = underpromotedtoi[move.promotion]
            return (64+3*(dv+1) + piece)+ 73*startcase
    elif not (dh == 0 or dv == 0 or abs(dh) == abs(dv)):
        dir = knight_directions.index((dh,dv))
        return (56+dir)+ 73*startcase
    else:
        return simplemove_to_index(move)

def index_to_move(index, board):
    startcase = index // 73
    moving = index % 73

    rank = chess.square_rank(startcase)
    file = chess.square_file(startcase)

    if moving < 56:
        dh, dv = DIRECTIONS[moving // 7]
        distance = moving % 7 + 1
        dh, dv = dh * distance, dv * distance
        promotion = None

    elif moving < 64:
        dh, dv = knight_directions[moving - 56]
        promotion = None

    else:
        rest = moving - 64
        dv = rest // 3 - 1
        promotion = itounderpromote[rest % 3]
        dh = 1 if board.turn == chess.WHITE else -1

    endcase = chess.square(file + dv, rank + dh)

    if promotion is None:
        piece = board.piece_at(startcase)
        last_rank = 7 if board.turn == chess.WHITE else 0
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(endcase) == last_rank
        ):
            promotion = chess.QUEEN

    return chess.Move(startcase, endcase, promotion=promotion)